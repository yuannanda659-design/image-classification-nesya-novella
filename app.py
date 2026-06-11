import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.set_page_config(
    page_title="Klasifikasi Anggur",
    page_icon="🍇",
    layout="centered"
)

st.title("🍇 Klasifikasi Anggur (Merah vs Hijau)")
st.write("Upload model (.tflite) dan gambar untuk melakukan klasifikasi buah.")

# Threshold untuk menolak gambar yang tidak dikenali
THRESHOLD = 0.80
class_names = ["Merah", "Hijau"]

@st.cache_resource
def load_model(model_path):
    # Ditambahkan safe_mode=False di sini
    return tf.keras.models.load_model(model_path, safe_mode=False)
    
# ==========================================
# TAMPILAN AWAL: INPUT MODEL & GAMBAR
# ==========================================
st.subheader("1️⃣ Upload Model")
uploaded_model = st.file_uploader(
    "Pilih file model (.tflite)",
    type=["tflite"]
)

st.subheader("2️⃣ Upload Gambar Uji")
uploaded_image = st.file_uploader(
    "Pilih gambar buah (.jpg/.jpeg/.png)",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# LOGIKA PROSES (Berjalan jika kedua file ada)
# ==========================================
if uploaded_model is not None and uploaded_image is not None:
    
    # 1. Simpan sementara & Deteksi tipe model
    file_extension = os.path.splitext(uploaded_model.name)[1].lower()
    temp_model_path = f"temp_model{file_extension}"
    
    with open(temp_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
        
    try:
        # Tentukan target size default
        target_size = (227, 227)
        model_type = ""

        # Load sesuai tipe model
        if file_extension == ".keras":
            model = tf.keras.models.load_model(temp_model_path)
            model_type = "keras"
            st.success("✅ Model Keras berhasil dimuat")
            
        elif file_extension == ".tflite":
            interpreter = tf.lite.Interpreter(model_path=temp_model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            model_type = "tflite"
            
            # Mengambil target size otomatis dari model TFLite [batch, height, width, channels]
            input_shape = input_details[0]['shape']
            target_size = (input_shape[1], input_shape[2])
            st.success(f"✅ Model TFLite berhasil dimuat")

        # 2. Pemrosesan Gambar
        img = Image.open(uploaded_image).convert("RGB")
        st.image(
            img,
            caption="Gambar Uji",
            use_container_width=True
        )

        # Resize gambar sesuai kebutuhan model
        img_resized = img.resize(target_size)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Proses Prediksi
        if model_type == "keras":
            hasil = model.predict(img_array, verbose=0)
            
        elif model_type == "tflite":
            # Sesuaikan tipe data dengan input TFLite (biasanya float32)
            if input_details[0]['dtype'] == np.float32:
                img_array = img_array.astype(np.float32)
                # Catatan: Jika model TFLite Anda membutuhkan normalisasi /255.0, 
                # hapus tanda pagar di bawah ini:
                # img_array = img_array / 255.0
                
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            hasil = interpreter.get_tensor(output_details[0]['index'])

        # 4. Pasca-Proses Hasil Prediksi
        prediksi = np.argmax(hasil)
        confidence = np.max(hasil)

        st.subheader("📊 Hasil Prediksi")

        if confidence < THRESHOLD:
            st.warning("⚠️ Saya tidak tahu (Di bawah threshold keyakinan)")
        else:
            st.success(
                f"✅ Buah terdeteksi sebagai: **{class_names[prediksi].upper()}**"
            )

        st.write(
            f"**Tingkat Keyakinan:** {confidence * 100:.2f}%"
        )

        st.subheader("Probabilitas Tiap Kelas")
        for i, nama in enumerate(class_names):
            st.write(
                f"**{nama.capitalize()}:** {hasil[0][i] * 100:.2f}%"
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses model atau gambar: {e}")

else:
    # Berikan panduan informasi di awal jika user belum upload salah satu / keduanya
    if uploaded_model is None:
        st.info("💡 Silakan upload file model (.keras atau .tflite) pada kolom pertama.")
    if uploaded_image is None:
        st.info("💡 Silakan upload gambar uji (.jpg/.png) pada kolom kedua.")

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# MNIST 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 데이터 형태 맞추기
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# 모델 정의 함수
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Multi-GPU 전략 설정
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # 모델 생성
    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, epochs=5, batch_size=64)

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc}')

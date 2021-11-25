#!/usr/bin/env python3
from __future__ import print_function
import tensorflow.keras
import theano

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print(tensorflow.keras.__version__)
print(theano.__version__)

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

#!/usr/bin/env python3
import mxnet as mx
a = mx.nd.ones((2, 3))
b = a * 2 + 1
b.asnumpy()

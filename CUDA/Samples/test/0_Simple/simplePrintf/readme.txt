Sample: simplePrintf
Minimum spec: SM 2.0

This CUDA Runtime API sample is a very basic sample that implements how to use the printf function in the device code. Specifically, for devices with compute capability less than 2.0, the function cuPrintf is called; otherwise, printf can be used directly.

Key concepts:
Debugging

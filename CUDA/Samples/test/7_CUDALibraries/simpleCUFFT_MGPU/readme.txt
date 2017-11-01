Sample: simpleCUFFT_MGPU
Minimum spec: SM 2.0

Example of using CUFFT. In this example, CUFFT is used to compute the 1D-convolution of some signal with some filter by transforming both into frequency domain, multiplying them together, and transforming the signal back to time domain on Multiple GPU.

Key concepts:
Image Processing
CUFFT Library

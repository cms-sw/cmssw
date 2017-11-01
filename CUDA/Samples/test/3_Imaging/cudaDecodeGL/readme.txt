Sample: cudaDecodeGL
Minimum spec: SM 2.0

This sample demonstrates how to efficiently use the CUDA Video Decoder API to decode video sources based on MPEG-2, VC-1, and H.264.  YUV to RGB conversion of video is accomplished with CUDA kernel.  The output result is rendered to a OpenGL surface.  The decoded video is black, but can be enabled with -displayvideo added to the command line.  Requires Compute Capability 2.0 or higher.

Key concepts:
Graphics Interop
Image Processing
Video Compression

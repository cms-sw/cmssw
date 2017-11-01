Sample: concurrentKernels
Minimum spec: SM 2.0

This sample demonstrates the use of CUDA streams for concurrent execution of several kernels on devices of compute capability 2.0 or higher.  Devices of compute capability 1.x will run the kernels sequentially.  It also illustrates how to introduce dependencies between CUDA streams with the new cudaStreamWaitEvent function introduced in CUDA 3.2

Key concepts:
Performance Strategies

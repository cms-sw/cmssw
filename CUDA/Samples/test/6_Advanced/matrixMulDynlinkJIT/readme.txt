Sample: matrixMulDynlinkJIT
Minimum spec: SM 2.0

This sample revisits matrix multiplication using the CUDA driver API. It demonstrates how to link to CUDA driver at runtime and how to use JIT (just-in-time) compilation from PTX code. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication. CUBLAS provides high-performance matrix multiplication.

Key concepts:
CUDA Driver API
CUDA Dynamically Linked Library

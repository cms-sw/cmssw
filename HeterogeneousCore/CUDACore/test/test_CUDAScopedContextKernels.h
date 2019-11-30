#ifndef HeterogeneousCore_CUDACore_test_CUDAScopedContextKernels_h
#define HeterogeneousCore_CUDACore_test_CUDAScopedContextKernels_h

#include <cuda_runtime.h>

void testCUDAScopedContextKernels_single(int *d, cudaStream_t stream);
void testCUDAScopedContextKernels_join(const int *d1, const int *d2, int *d3, cudaStream_t stream);

#endif

#ifndef HeterogeneousCore_CUDACore_test_CUDAScopedContextKernels_h
#define HeterogeneousCore_CUDACore_test_CUDAScopedContextKernels_h

#include <cuda/api_wrappers.h>

void testCUDAScopedContextKernels_single(int *d, cuda::stream_t<>& stream);
void testCUDAScopedContextKernels_join(const int *d1, const int *d2, int *d3, cuda::stream_t<>& stream);

#endif

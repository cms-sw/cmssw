#ifndef HeterogeneousCore_CUDACore_test_ScopedContextKernels_h
#define HeterogeneousCore_CUDACore_test_ScopedContextKernels_h

#include <cuda_runtime.h>

namespace cms {
  namespace cudatest {
    void testScopedContextKernels_single(int *d, cudaStream_t stream);
    void testScopedContextKernels_join(const int *d1, const int *d2, int *d3, cudaStream_t stream);
  }  // namespace cudatest
}  // namespace cms

#endif

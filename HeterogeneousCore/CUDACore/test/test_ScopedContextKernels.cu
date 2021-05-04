#include "test_ScopedContextKernels.h"

namespace {
  __global__ void single_mul(int *d) { d[0] = d[0] * 2; }

  __global__ void join_add(const int *d1, const int *d2, int *d3) { d3[0] = d1[0] + d2[0]; }
}  // namespace

namespace cms {
  namespace cudatest {
    void testScopedContextKernels_single(int *d, cudaStream_t stream) { single_mul<<<1, 1, 0, stream>>>(d); }

    void testScopedContextKernels_join(const int *d1, const int *d2, int *d3, cudaStream_t stream) {
      join_add<<<1, 1, 0, stream>>>(d1, d2, d3);
    }
  }  // namespace cudatest
}  // namespace cms

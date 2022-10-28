#include <cuda_runtime.h>

#include "CUDADataFormats/PortableTestObjects/interface/TestDeviceCollection.h"

#include "TestAlgo.h"

namespace cudatest {

  static __global__ void testAlgoKernel(cudatest::TestDeviceCollection::View view, int32_t size) {
    const int32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;
    const cudatest::Matrix matrix{{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}, {3, 6, 9, 12, 15, 18}};

    if (thread == 0) {
      view.r() = 1.;
    }
    for (auto i = thread; i < size; i += stride) {
      view[i] = {0., 0., 0., i, matrix * i};
    }
  }

  void TestAlgo::fill(cudatest::TestDeviceCollection& collection, cudaStream_t stream) const {
    const uint32_t maxThreadsPerBlock = 1024;

    uint32_t threadsPerBlock = maxThreadsPerBlock;
    uint32_t blocksPerGrid = (collection->metadata().size() + threadsPerBlock - 1) / threadsPerBlock;

    testAlgoKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(collection.view(), collection->metadata().size());
  }

}  // namespace cudatest

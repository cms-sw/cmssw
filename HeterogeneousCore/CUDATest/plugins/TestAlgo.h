#ifndef HeterogeneousCore_CUDATest_plugins_TestAlgo_h
#define HeterogeneousCore_CUDATest_plugins_TestAlgo_h

#include "CUDADataFormats/PortableTestObjects/interface/TestDeviceCollection.h"
#include "CUDADataFormats/PortableTestObjects/interface/TestHostCollection.h"

namespace cudatest {

  class TestAlgo {
  public:
    void fill(cudatest::TestDeviceCollection& collection, cudaStream_t stream) const;
    void fill(cudatest::TestHostCollection& collection) const;
  };

}  // namespace cudatest

#endif  // HeterogeneousCore_CUDATest_plugins_TestAlgo_h

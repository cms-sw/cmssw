// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include <alpaka/alpaka.hpp>

#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TestAlgoKernel {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, int32_t* id, int32_t size) const {
      int32_t idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
      if (idx < size) {
        id[idx] = idx;
      }
    }
  };

  void TestAlgo::fill(Queue& queue, portabletest::TestDeviceCollection& collection) const {
    auto const& deviceProperties = alpaka::getAccDevProps<Acc1D>(alpaka::getDev(queue));
    uint32_t maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

    uint32_t threadsPerBlock = maxThreadsPerBlock;
    uint32_t blocksPerGrid = (collection->size() + threadsPerBlock - 1) / threadsPerBlock;
    uint32_t elementsPerThread = 1;
    auto workDiv = WorkDiv1D{blocksPerGrid, threadsPerBlock, elementsPerThread};

    alpaka::exec<Acc1D>(queue, workDiv, TestAlgoKernel{}, &collection->id(0), collection->size());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

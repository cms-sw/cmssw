#ifndef HeterogeneousCore_AlpakaTest_plugins_alpaka_testPtrAlgo_h
#define HeterogeneousCore_AlpakaTest_plugins_alpaka_testPtrAlgo_h

#include "DataFormats/PortableTestObjects/interface/TestProductWithPtr.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  portabletest::TestProductWithPtr<Device> testPtrAlgoAsync(Queue& queue, int size);
}

#endif

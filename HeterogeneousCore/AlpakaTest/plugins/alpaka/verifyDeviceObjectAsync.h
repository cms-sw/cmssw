#ifndef HeterogeneousCore_AlpakaTest_plugins_alpaka_verifyDeviceObjectAsync_h
#define HeterogeneousCore_AlpakaTest_plugins_alpaka_verifyDeviceObjectAsync_h

#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  cms::alpakatools::host_buffer<bool> verifyDeviceObjectAsync(Queue& queue,
                                                              portabletest::TestDeviceObject const& deviceObject);
}

#endif

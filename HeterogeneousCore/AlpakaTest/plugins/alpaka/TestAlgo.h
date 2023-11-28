#ifndef HeterogeneousCore_AlpakaTest_plugins_alpaka_TestAlgo_h
#define HeterogeneousCore_AlpakaTest_plugins_alpaka_TestAlgo_h

#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TestAlgo {
  public:
    void fill(Queue& queue, portabletest::TestDeviceCollection& collection, double xvalue = 0.) const;
    void fillObject(
        Queue& queue, portabletest::TestDeviceObject& object, double x, double y, double z, int32_t id) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HeterogeneousCore_AlpakaTest_plugins_alpaka_TestAlgo_h

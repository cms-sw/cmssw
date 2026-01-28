#ifndef HeterogeneousCore_AlpakaTest_plugins_alpaka_TestAlgo_h
#define HeterogeneousCore_AlpakaTest_plugins_alpaka_TestAlgo_h

#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TestAlgo {
  public:
    void fill(Queue& queue, portabletest::TestDeviceCollection& collection, double xvalue = 0.) const;
    void fillObject(
        Queue& queue, portabletest::TestDeviceObject& object, double x, double y, double z, int32_t id) const;

    portabletest::TestDeviceCollection update(Queue& queue,
                                              portabletest::TestDeviceCollection const& input,
                                              AlpakaESTestDataEDevice const& esData) const;

    portabletest::TestDeviceCollection2 updateMulti2(Queue& queue,
                                                     portabletest::TestDeviceCollection2 const& input,
                                                     AlpakaESTestDataEDevice const& esData) const;
    portabletest::TestDeviceCollection3 updateMulti3(Queue& queue,
                                                     portabletest::TestDeviceCollection3 const& input,
                                                     AlpakaESTestDataEDevice const& esData) const;

    struct UpdateInfo {
      int x, y, z;
    };
    portabletest::TestDeviceCollection update(Queue& queue,
                                              portabletest::TestDeviceCollection const& input,
                                              UpdateInfo const* d_updateInfo) const;
    portabletest::TestDeviceCollection2 updateMulti2(Queue& queue,
                                                     portabletest::TestDeviceCollection2 const& input,
                                                     UpdateInfo const* d_updateInfo) const;
    portabletest::TestDeviceCollection3 updateMulti3(Queue& queue,
                                                     portabletest::TestDeviceCollection3 const& input,
                                                     UpdateInfo const* d_updateInfo) const;

    void fillMulti2(Queue& queue, portabletest::TestDeviceCollection2& collection, double xvalue = 0.) const;
    void fillMulti3(Queue& queue, portabletest::TestDeviceCollection3& collection, double xvalue = 0.) const;

    void checkZero(Queue& queue, portabletest::TestDeviceCollection const& collection) const;
    void checkZero(Queue& queue, portabletest::TestDeviceCollection2 const& collection) const;
    void checkZero(Queue& queue, portabletest::TestDeviceCollection3 const& collection) const;
    void checkZero(Queue& queue, portabletest::TestDeviceObject const& object) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HeterogeneousCore_AlpakaTest_plugins_alpaka_TestAlgo_h

#include <alpaka/alpaka.hpp>
#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TestTorchDeviceMatchAlpaka : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(TestTorchDeviceMatchAlpaka);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST_SUITE_END();

  public:
    void test();
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestTorchDeviceMatchAlpaka);

  void TestTorchDeviceMatchAlpaka::test() {
    // alpaka setup
    Platform platform;
    const auto& devices = alpaka::getDevs(platform);

    std::cout << "Devices:" << std::endl;
    for (auto& device : devices) {
      std::cout << "- " << alpaka::getName(device) << std::endl;
      auto torch_device = cms::torch::alpaka::device(device);
      CPPUNIT_ASSERT(torch_device.type() == cms::torch::alpaka::kTorchDeviceType);
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include <torch/torch.h>
#include <vector>

#include <cppunit/extensions/HelperMacros.h>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/test/testUtilities.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::cms::torch::alpaka;

  class TestModelAlpakaNamespace : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(TestModelAlpakaNamespace);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST_SUITE_END();

  public:
    void test();
    const int64_t batch_size_ = 2 << 10;
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestModelAlpakaNamespace);

  void TestModelAlpakaNamespace::test() {
    // alpaka setup
    Platform platform;
    std::vector<Device> devices = ::alpaka::getDevs(platform);
    CPPUNIT_ASSERT(!devices.empty());
    const auto& alpaka_device = devices[0];
    Queue queue{alpaka_device};

    auto device = cms::torch::alpaka::device(queue);
    std::cout << "Device: " << device << std::endl;
    auto inputs = ::torch::ones({batch_size_, 3}, device);

    ClassifierModel model;
    model.to(device);
    auto outputs = model.forward(inputs);
    CPPUNIT_ASSERT(::torch::allclose(outputs, ::torch::full_like(outputs, 0.5f)));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

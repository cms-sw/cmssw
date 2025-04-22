#include <alpaka/alpaka.hpp>
#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/interface/Model.h"
#include "PhysicsTools/PyTorch/test/testUtilities.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::cms::torch::alpaka;

  class TestModelJIT : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(TestModelJIT);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST(testMultiBranch);
    CPPUNIT_TEST_SUITE_END();

  public:
    void test();
    void testMultiBranch();
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestModelJIT);

  void TestModelJIT::test() {
    // alpaka setup
    Platform platform;
    std::vector<Device> devices = ::alpaka::getDevs(platform);
    CPPUNIT_ASSERT(!devices.empty());
    const auto& device = devices[0];
    Queue queue{device};

    const std::size_t batch_size = 2 << 10;
    std::vector<::torch::IValue> inputs;
    inputs.push_back(::torch::ones({batch_size, 3}, cms::torch::alpaka::device(queue)));

    auto m_path = get_path("/src/PhysicsTools/PyTorch/models/jit_classification_model.pt");
    Model<CompilationType::kJustInTime> jit_model(m_path);
    jit_model.to(queue);
    std::cout << "Device: " << jit_model.device() << std::endl;
    CPPUNIT_ASSERT(cms::torch::alpaka::device(queue) == jit_model.device());
    auto outputs = jit_model.forward(inputs);
    CPPUNIT_ASSERT(::torch::allclose(outputs.toTensor(), ::torch::full_like(outputs.toTensor(), 0.5f)));
  }

  void TestModelJIT::testMultiBranch() {
    // alpaka setup
    Platform platform;
    std::vector<Device> devices = ::alpaka::getDevs(platform);
    CPPUNIT_ASSERT(!devices.empty());
    const auto& device = devices[0];
    Queue queue{device};

    const std::size_t batch_size = 2 << 10;
    std::vector<::torch::IValue> inputs;
    inputs.push_back(::torch::ones({batch_size, 5}, cms::torch::alpaka::device(queue)));

    auto m_path = get_path("/src/PhysicsTools/PyTorch/models/jit_multi_branch_model.pt");
    Model<CompilationType::kJustInTime> jit_model(m_path);
    jit_model.to(queue);
    std::cout << "Device: " << jit_model.device() << std::endl;
    CPPUNIT_ASSERT(cms::torch::alpaka::device(queue) == jit_model.device());
    auto outputs = jit_model.forward(inputs).toTuple();

    auto class_probs = outputs->elements()[0].toTensor();
    auto reg_value = outputs->elements()[1].toTensor();

    CPPUNIT_ASSERT(::torch::allclose(class_probs, ::torch::full_like(class_probs, 0.2f)));
    CPPUNIT_ASSERT(::torch::allclose(reg_value, ::torch::full_like(reg_value, 15.7286f)));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include <alpaka/alpaka.hpp>
#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/interface/Model.h"
#include "PhysicsTools/PyTorch/test/testUtilities.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::cms::torch::alpaka;

  class TestModelAOT : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(TestModelAOT);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST_SUITE_END();

  public:
    void test();
    std::string shared_lib();
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestModelAOT);

  std::string TestModelAOT::shared_lib() {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    return get_path("/src/PhysicsTools/PyTorch/models/aot_regression_model_cuda_el9_amd64_gcc12.pt2");
#elif ALPAKA_ACC_GPU_HIP_ENABLED
    std::cerr << "FAILED: ROCm backend not yet supported, see: "
                 "https://github.com/pytorch/pytorch/blob/main/aten/CMakeLists.txt#L75"
              << std::endl;
    return "";
#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    return get_path("/src/PhysicsTools/PyTorch/models/aot_regression_model_cpu_el9_amd64_gcc12.pt2");
#else
    std::cerr << "FAILED: Unable to detect backend type." << std::endl;
    return "";
#endif
  }

  void TestModelAOT::test() {
    // alpaka setup
    Platform platform;
    std::vector<Device> devices = ::alpaka::getDevs(platform);
    CPPUNIT_ASSERT(!devices.empty());
    const auto& device = devices[0];
    Queue queue{device};

    const std::size_t batch_size = 2 << 10;
    std::vector<::torch::IValue> inputs;
    inputs.push_back(torch::ones({batch_size, 3}, cms::torch::alpaka::device(queue)));

    std::vector<at::Tensor> inputs_tensor;
    for (const auto& val : inputs)
      inputs_tensor.push_back(val.toTensor());

    auto lib_path = shared_lib();
    CPPUNIT_ASSERT_MESSAGE("FAILED: Architecture compiled shared library missing.", !lib_path.empty());

    Model<CompilationType::kAheadOfTime> aot_model(lib_path);
    aot_model.to(queue);
    std::cout << "Device: " << aot_model.device() << std::endl;
    CPPUNIT_ASSERT(cms::torch::alpaka::device(queue) == aot_model.device());
    auto outputs = aot_model.forward(inputs_tensor);
    for (const auto& val : outputs) {
      CPPUNIT_ASSERT(::torch::allclose(val, ::torch::full_like(val, 0.5f)));
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "PhysicsTools/PyTorch/interface/TorchInterface.h"
#include "PhysicsTools/PyTorch/interface/ScriptModuleLoad.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

namespace torchtest {

  constexpr auto modelPath = "PhysicsTools/PyTorch/data/linear_dnn.pt";

  class TestJitLoad : public CppUnit::TestFixture {
  public:
    void testJitLoadNoException();
    void testJitLoadThrowException();
    void testJitLoadToDirectDevice();

  private:
    CPPUNIT_TEST_SUITE(TestJitLoad);
    CPPUNIT_TEST(testJitLoadNoException);
    CPPUNIT_TEST(testJitLoadThrowException);
    CPPUNIT_TEST(testJitLoadToDirectDevice);
    CPPUNIT_TEST_SUITE_END();

    const int64_t batch_size_ = 8;

    template <typename Fn>
    void forEachCudaDevice(Fn&& fn) {
      int count = ::torch::cuda::device_count();
      auto m_path = edm::FileInPath(modelPath).fullPath();
      for (int i = 0; i < count; ++i) {
        auto dev = ::torch::Device(::torch::kCUDA, i);
        std::cout << "Running test on device " << dev << std::endl;
        fn(dev, m_path);
      }
    }
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestJitLoad);

  void TestJitLoad::testJitLoadNoException() {
    auto m_path = edm::FileInPath(modelPath).fullPath();
    const auto model = cms::torch::load(m_path);
  }

  void TestJitLoad::testJitLoadThrowException() {
    CPPUNIT_ASSERT_THROW(cms::torch::load("/non_existing_model.pt"), cms::Exception);
  }

  /**
   * @brief Test loading a model to a specific device directly without CPU intermediary
   * @note torchlib interface does not support async loading, use model.to(device, true) to load asynchronously 
   */
  void TestJitLoad::testJitLoadToDirectDevice() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = edm::FileInPath(modelPath).fullPath();
    forEachCudaDevice([&](auto dev, auto m_path) {
      auto m = cms::torch::load(m_path, dev);

      auto inputs = std::vector<c10::IValue>();
      inputs.push_back(torch::ones({batch_size_, 3}, dev));
      auto outputs = m.forward(inputs).toTensor();

      auto expected = torch::tensor({2.1f, 1.8f}, torch::TensorOptions().device(dev)).repeat({batch_size_, 1});
      CPPUNIT_ASSERT(torch::allclose(outputs, expected));
    });
  }

}  // namespace torchtest

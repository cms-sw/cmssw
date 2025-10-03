#include "PhysicsTools/PyTorch/interface/TorchCompat.h"
#include "PhysicsTools/PyTorch/interface/ScriptModuleLoad.h"
#include "PhysicsTools/PyTorch/test/testTorchBase.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

namespace torchtest {

  class TestJitLoad : public testTorchBase {
  public:
    std::string script() const override;
    void testJitLoadNoException();
    void testJitLoadThrowException();
    void testJitLoadToDirectDevice();

  private:
    CPPUNIT_TEST_SUITE(TestJitLoad);
    CPPUNIT_TEST(testJitLoadNoException);
    CPPUNIT_TEST(testJitLoadThrowException);
    CPPUNIT_TEST(testJitLoadToDirectDevice);
    CPPUNIT_TEST_SUITE_END();

    const int64_t batch_size_ = 2 << 10;
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestJitLoad);

  std::string TestJitLoad::script() const { return "testExportLinearDnn.py"; }

  void TestJitLoad::testJitLoadNoException() {
    auto model_path = modelPath() + "/linear_dnn.pt";
    const auto model = cms::torch::load(model_path);
  }

  void TestJitLoad::testJitLoadThrowException() {
    auto model_path = modelPath() + "/non_existing_model.pt";
    CPPUNIT_ASSERT_THROW(cms::torch::load(model_path), cms::Exception);
  }

  /**
   * @brief Test loading a model to a specific device directly without CPU intermediary
   * @note torchlib interface does not support async loading, use model.to(device, true) to load asynchronously 
   */
  void TestJitLoad::testJitLoadToDirectDevice() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = modelPath() + "/linear_dnn.pt";
    const auto dev = torch::Device(torch::kCUDA, 0);
    auto m = cms::torch::load(m_path, dev);

    auto inputs = std::vector<c10::IValue>();
    inputs.push_back(torch::ones({batch_size_, 3}, dev));
    auto outputs = m.forward(inputs).toTensor();

    auto expected = torch::tensor({2.1f, 1.8f}, torch::TensorOptions().device(dev)).repeat({batch_size_, 1});
    CPPUNIT_ASSERT(torch::allclose(outputs, expected));
  }

}  // namespace torchtest
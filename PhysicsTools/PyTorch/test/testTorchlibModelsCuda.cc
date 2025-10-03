#include <vector>

#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "PhysicsTools/PyTorch/test/testTorchlibModels.h"

namespace torchtest {

  class TestTorchlibModelsCuda : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(TestTorchlibModelsCuda);
    CPPUNIT_TEST(testClassificationCuda);
    CPPUNIT_TEST(testRegressionCuda);
    CPPUNIT_TEST(testMultiTaskModelCuda);
    CPPUNIT_TEST_SUITE_END();

  public:
    void testClassificationCuda();
    void testRegressionCuda();
    void testMultiTaskModelCuda();

    const int64_t batch_size_ = 2 << 10;
    const torch::Device device_ = torch::Device(torch::kCUDA, 0);
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestTorchlibModelsCuda);

  void TestTorchlibModelsCuda::testClassificationCuda() {
    // disable test on non-CUDA devices
    if (not cms::cudatest::testDevices())
      return;

    ClassifierModel model;
    model.to(device_);

    auto inputs = torch::ones({batch_size_, 3}, device_);
    auto outputs = model.forward(inputs);

    CPPUNIT_ASSERT(torch::allclose(outputs, torch::full_like(outputs, 0.5f)));
  }

  void TestTorchlibModelsCuda::testRegressionCuda() {
    // disable test on non-CUDA devices
    if (not cms::cudatest::testDevices())
      return;

    RegressionModel model;
    model.to(device_);

    auto inputs = torch::ones({batch_size_, 3}, device_);
    auto outputs = model.forward(inputs);

    CPPUNIT_ASSERT(torch::allclose(outputs, torch::full_like(outputs, 0.5f)));
  }

  void TestTorchlibModelsCuda::testMultiTaskModelCuda() {
    // disable test on non-CUDA devices
    if (not cms::cudatest::testDevices())
      return;

    MultiTaskModel model;
    model.to(device_);

    auto inputs = torch::ones({batch_size_, 5}, device_);
    auto [class_probs, reg_output] = model.forward(inputs);

    CPPUNIT_ASSERT(torch::allclose(class_probs, torch::full_like(class_probs, 0.2f)));
    CPPUNIT_ASSERT(torch::allclose(reg_output, torch::full_like(reg_output, 15.7286f)));
  }

}  // namespace torchtest
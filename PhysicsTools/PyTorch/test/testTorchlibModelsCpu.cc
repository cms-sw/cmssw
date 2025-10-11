#include <vector>

#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/PyTorch/test/testTorchlibModels.h"

namespace torchtest {

  class TestTorchlibModelsCpu : public CppUnit::TestFixture {
  private:
    CPPUNIT_TEST_SUITE(TestTorchlibModelsCpu);
    CPPUNIT_TEST(testClassificationCpu);
    CPPUNIT_TEST(testRegressionCpu);
    CPPUNIT_TEST(testMultiTaskModelCpu);
    CPPUNIT_TEST_SUITE_END();

  public:
    void testClassificationCpu();
    void testRegressionCpu();
    void testMultiTaskModelCpu();

    const int64_t batch_size_ = 8;
    const torch::Device device_ = torch::Device(torch::kCPU, 0);
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestTorchlibModelsCpu);

  void TestTorchlibModelsCpu::testClassificationCpu() {
    ClassifierModel model;

    auto inputs = torch::ones({batch_size_, 3}, device_);
    auto outputs = model.forward(inputs);

    CPPUNIT_ASSERT(torch::allclose(outputs, torch::full_like(outputs, 0.5f)));
  }

  void TestTorchlibModelsCpu::testRegressionCpu() {
    RegressionModel model;

    auto inputs = torch::ones({batch_size_, 3}, device_);
    auto outputs = model.forward(inputs);

    CPPUNIT_ASSERT(torch::allclose(outputs, torch::full_like(outputs, 0.5f)));
  }

  void TestTorchlibModelsCpu::testMultiTaskModelCpu() {
    MultiTaskModel model;

    auto inputs = torch::ones({batch_size_, 5}, device_);
    auto [class_probs, reg_output] = model.forward(inputs);

    CPPUNIT_ASSERT(torch::allclose(class_probs, torch::full_like(class_probs, 0.2f)));
    CPPUNIT_ASSERT(torch::allclose(reg_output, torch::full_like(reg_output, 15.7286f)));
  }

}  // namespace torchtest

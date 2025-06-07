#include <torch/torch.h>
#include <vector>

#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/PyTorch/test/testUtilities.h"

class TestModelCpu : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestModelCpu);
  CPPUNIT_TEST(testClassificationCpu);
  CPPUNIT_TEST(testRegressionCpu);
  CPPUNIT_TEST(testMultiTaskModelCpu);
  CPPUNIT_TEST_SUITE_END();

public:
  void testClassificationCpu();
  void testRegressionCpu();
  void testMultiTaskModelCpu();

  const int64_t batch_size_ = 2 << 10;
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestModelCpu);

void TestModelCpu::testClassificationCpu() {
  auto device = torch::Device(torch::kCPU, 0);
  auto inputs = torch::ones({batch_size_, 3}, device);

  ClassifierModel model;
  auto outputs = model.forward(inputs);
  CPPUNIT_ASSERT(torch::allclose(outputs, torch::full_like(outputs, 0.5f)));
}

void TestModelCpu::testRegressionCpu() {
  auto device = torch::Device(torch::kCPU, 0);
  auto inputs = torch::ones({batch_size_, 3}, device);

  RegressionModel model;
  auto outputs = model.forward(inputs);
  CPPUNIT_ASSERT(torch::allclose(outputs, torch::full_like(outputs, 0.5f)));
}

void TestModelCpu::testMultiTaskModelCpu() {
  auto device = torch::Device(torch::kCPU, 0);
  auto inputs = torch::ones({batch_size_, 5}, device);

  MultiTaskModel model;
  auto [class_probs, reg_output] = model.forward(inputs);
  CPPUNIT_ASSERT(torch::allclose(class_probs, torch::full_like(class_probs, 0.2f)));
  CPPUNIT_ASSERT(torch::allclose(reg_output, torch::full_like(reg_output, 15.7286f)));
}

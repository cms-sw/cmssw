#include <torch/torch.h>
#include <vector>

#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/PyTorch/test/testUtilities.h"

class TestModel : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestModel);
  CPPUNIT_TEST(testClassificationCpu);
  CPPUNIT_TEST(testClassificationCuda);
  CPPUNIT_TEST(testRegressionCpu);
  CPPUNIT_TEST(testRegressionCuda);
  CPPUNIT_TEST(testMultiTaskModelCpu);
  CPPUNIT_TEST(testMultiTaskModelCuda);
  CPPUNIT_TEST_SUITE_END();

public:
  void testClassificationCpu();
  void testClassificationCuda();
  void testRegressionCpu();
  void testRegressionCuda();
  void testMultiTaskModelCpu();
  void testMultiTaskModelCuda();

  const int64_t batch_size_ = 2 << 10;
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestModel);

void TestModel::testClassificationCpu() {
  auto device = torch::Device(torch::kCPU, 0);
  auto inputs = torch::ones({batch_size_, 3}, device = device);

  ClassifierModel model;
  auto outputs = model.forward(inputs);
  CPPUNIT_ASSERT(torch::allclose(outputs, torch::full_like(outputs, 0.5f)));
}

void TestModel::testClassificationCuda() {
  auto device = torch::Device(torch::kCUDA, 0);
  auto inputs = torch::ones({batch_size_, 3}, device = device);

  ClassifierModel model;
  model.to(device);
  auto outputs = model.forward(inputs);
  CPPUNIT_ASSERT(torch::allclose(outputs, torch::full_like(outputs, 0.5f)));
}

void TestModel::testRegressionCpu() {
  auto device = torch::Device(torch::kCPU, 0);
  auto inputs = torch::ones({batch_size_, 3}, device = device);

  RegressionModel model;
  auto outputs = model.forward(inputs);
  CPPUNIT_ASSERT(torch::allclose(outputs, torch::full_like(outputs, 0.5f)));
}

void TestModel::testRegressionCuda() {
  auto device = torch::Device(torch::kCUDA, 0);
  auto inputs = torch::ones({batch_size_, 3}, device = device);

  RegressionModel model;
  model.to(device);
  auto outputs = model.forward(inputs);
  CPPUNIT_ASSERT(torch::allclose(outputs, torch::full_like(outputs, 0.5f)));
}

void TestModel::testMultiTaskModelCpu() {
  auto device = torch::Device(torch::kCPU, 0);
  auto inputs = torch::ones({batch_size_, 5}, device = device);

  MultiTaskModel model;
  auto [class_probs, reg_output] = model.forward(inputs);
  CPPUNIT_ASSERT(torch::allclose(class_probs, torch::full_like(class_probs, 0.2f)));
  CPPUNIT_ASSERT(torch::allclose(reg_output, torch::full_like(reg_output, 15.7286f)));
}

void TestModel::testMultiTaskModelCuda() {
  auto device = torch::Device(torch::kCUDA, 0);
  auto inputs = torch::ones({batch_size_, 5}, device = device);

  MultiTaskModel model;
  model.to(device);
  auto [class_probs, reg_output] = model.forward(inputs);
  CPPUNIT_ASSERT(torch::allclose(class_probs, torch::full_like(class_probs, 0.2f)));
  CPPUNIT_ASSERT(torch::allclose(reg_output, torch::full_like(reg_output, 15.7286f)));
}

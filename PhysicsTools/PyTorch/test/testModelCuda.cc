#include <torch/torch.h>
#include <vector>

#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "PhysicsTools/PyTorch/test/testUtilities.h"

class TestModelCuda : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestModelCuda);
  CPPUNIT_TEST(testClassificationCuda);
  CPPUNIT_TEST(testRegressionCuda);
  CPPUNIT_TEST(testMultiTaskModelCuda);
  CPPUNIT_TEST_SUITE_END();

public:
  void testClassificationCuda();
  void testRegressionCuda();
  void testMultiTaskModelCuda();

  const int64_t batch_size_ = 2 << 10;
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestModelCuda);

void TestModelCuda::testClassificationCuda() {
  // temporary workaround to disable test on non-CUDA devices
  if (not cms::cudatest::testDevices())
    return;

  auto device = torch::Device(torch::kCUDA, 0);
  auto inputs = torch::ones({batch_size_, 3}, device);

  ClassifierModel model;
  model.to(device);
  auto outputs = model.forward(inputs);
  CPPUNIT_ASSERT(torch::allclose(outputs, torch::full_like(outputs, 0.5f)));
}

void TestModelCuda::testRegressionCuda() {
  // temporary workaround to disable test on non-CUDA devices
  if (not cms::cudatest::testDevices())
    return;

  auto device = torch::Device(torch::kCUDA, 0);
  auto inputs = torch::ones({batch_size_, 3}, device);

  RegressionModel model;
  model.to(device);
  auto outputs = model.forward(inputs);
  CPPUNIT_ASSERT(torch::allclose(outputs, torch::full_like(outputs, 0.5f)));
}

void TestModelCuda::testMultiTaskModelCuda() {
  // temporary workaround to disable test on non-CUDA devices
  if (not cms::cudatest::testDevices())
    return;

  auto device = torch::Device(torch::kCUDA, 0);
  auto inputs = torch::ones({batch_size_, 5}, device);

  MultiTaskModel model;
  model.to(device);
  auto [class_probs, reg_output] = model.forward(inputs);
  CPPUNIT_ASSERT(torch::allclose(class_probs, torch::full_like(class_probs, 0.2f)));
  CPPUNIT_ASSERT(torch::allclose(reg_output, torch::full_like(reg_output, 15.7286f)));
}

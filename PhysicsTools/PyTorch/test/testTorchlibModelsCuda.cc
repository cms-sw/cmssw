#include <vector>

#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "PhysicsTools/PyTorch/test/testTorchlibModels.h"

namespace torchtest {

  template <typename Fn>
  void forEachCudaDevice(Fn&& fn) {
    int count = ::torch::cuda::device_count();
    for (int i = 0; i < count; ++i) {
      auto dev = ::torch::Device(::torch::kCUDA, i);
      std::cout << "Running test on device " << dev << std::endl;
      fn(dev);
    }
  }

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

    const int64_t batch_size_ = 8;
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestTorchlibModelsCuda);

  void TestTorchlibModelsCuda::testClassificationCuda() {
    // disable test on non-CUDA devices
    if (not cms::cudatest::testDevices())
      return;

    forEachCudaDevice([&](auto dev) {
      ClassifierModel model;
      model.to(dev);

      auto inputs = torch::ones({batch_size_, 3}, dev);
      auto outputs = model.forward(inputs);

      CPPUNIT_ASSERT(torch::allclose(outputs, torch::full_like(outputs, 0.5f)));
    });
  }

  void TestTorchlibModelsCuda::testRegressionCuda() {
    // disable test on non-CUDA devices
    if (not cms::cudatest::testDevices())
      return;

    forEachCudaDevice([&](auto dev) {
      RegressionModel model;
      model.to(dev);

      auto inputs = torch::ones({batch_size_, 3}, dev);
      auto outputs = model.forward(inputs);

      CPPUNIT_ASSERT(torch::allclose(outputs, torch::full_like(outputs, 0.5f)));
    });
  }

  void TestTorchlibModelsCuda::testMultiTaskModelCuda() {
    // disable test on non-CUDA devices
    if (not cms::cudatest::testDevices())
      return;

    forEachCudaDevice([&](auto dev) {
      MultiTaskModel model;
      model.to(dev);

      auto inputs = torch::ones({batch_size_, 5}, dev);
      auto [class_probs, reg_output] = model.forward(inputs);

      CPPUNIT_ASSERT(torch::allclose(class_probs, torch::full_like(class_probs, 0.2f)));
      CPPUNIT_ASSERT(torch::allclose(reg_output, torch::full_like(reg_output, 15.7286f)));
    });
  }

}  // namespace torchtest

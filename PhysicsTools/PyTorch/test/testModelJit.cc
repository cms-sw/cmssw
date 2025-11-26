#include <cppunit/extensions/HelperMacros.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "PhysicsTools/PyTorch/interface/Model.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

namespace torchtest {

  constexpr auto modelPath = "PhysicsTools/PyTorch/data/linear_dnn.pt";

  template <typename Fn>
  void forEachCudaDevice(Fn&& fn) {
    int count = ::torch::cuda::device_count();
    for (int i = 0; i < count; ++i) {
      auto dev = ::torch::Device(::torch::kCUDA, i);
      std::cout << "Running test on device " << dev << std::endl;
      fn(dev);
    }
  }

  class TestModelJIT : public CppUnit::TestFixture {
  public:
    void testCtor_DefaultDeviceIsCpu();
    void testCtor_ExplicitDeviceIsHonored();
    void testCtor_BadModelPathThrows();
    void testToDevice_UpdatesUnderlyingState();
    void testToDevice_NonBlocking();
    void testForward_IdempotentOutput();
    void testForward_OutputOnCorrectDevice();
    void testAsyncExecution();

  private:
    CPPUNIT_TEST_SUITE(TestModelJIT);
    CPPUNIT_TEST(testCtor_DefaultDeviceIsCpu);
    CPPUNIT_TEST(testCtor_ExplicitDeviceIsHonored);
    CPPUNIT_TEST(testCtor_BadModelPathThrows);
    CPPUNIT_TEST(testToDevice_UpdatesUnderlyingState);
    CPPUNIT_TEST(testToDevice_NonBlocking);
    CPPUNIT_TEST(testForward_IdempotentOutput);
    CPPUNIT_TEST(testForward_OutputOnCorrectDevice);
    CPPUNIT_TEST(testAsyncExecution);
    CPPUNIT_TEST_SUITE_END();

    const int64_t batch_size_ = 8;
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestModelJIT);

  void TestModelJIT::testCtor_DefaultDeviceIsCpu() {
    auto m_path = edm::FileInPath(modelPath).fullPath();
    auto m = cms::torch::Model(m_path);

    CPPUNIT_ASSERT_EQUAL(::torch::kCPU, m.device().type());
  }

  void TestModelJIT::testCtor_ExplicitDeviceIsHonored() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = edm::FileInPath(modelPath).fullPath();
    forEachCudaDevice([&](auto dev) {
      auto m = cms::torch::Model(m_path, dev);
      CPPUNIT_ASSERT_EQUAL(dev, m.device());
    });
  }

  void TestModelJIT::testCtor_BadModelPathThrows() {
    CPPUNIT_ASSERT_THROW(cms::torch::Model m("/not_existing_model.pt"), cms::Exception);
  }

  void TestModelJIT::testToDevice_UpdatesUnderlyingState() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = edm::FileInPath(modelPath).fullPath();
    forEachCudaDevice([&](auto dev) {
      auto m = cms::torch::Model(m_path);
      m.to(dev);

      CPPUNIT_ASSERT_EQUAL(dev, m.device());

      m.to(::torch::kCPU);
      CPPUNIT_ASSERT_EQUAL(::torch::kCPU, m.device().type());
    });
  }

  void TestModelJIT::testToDevice_NonBlocking() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = edm::FileInPath(modelPath).fullPath();
    forEachCudaDevice([&](auto dev) {
      auto m = cms::torch::Model(m_path);
      m.to(dev, true);

      CPPUNIT_ASSERT_EQUAL(dev, m.device());
    });
  }

  void TestModelJIT::testForward_IdempotentOutput() {
    auto m_path = edm::FileInPath(modelPath).fullPath();
    forEachCudaDevice([&](auto dev) {
      auto m = cms::torch::Model(m_path);
      auto inputs = std::vector<torch::IValue>();
      inputs.push_back(torch::randn({batch_size_, 3}));
      auto out1 = m.forward(inputs).toTensor();
      auto out2 = m.forward(inputs).toTensor();
      CPPUNIT_ASSERT(out1.equal(out2));
    });
  }

  void TestModelJIT::testForward_OutputOnCorrectDevice() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = edm::FileInPath(modelPath).fullPath();
    forEachCudaDevice([&](auto dev) {
      auto m = cms::torch::Model(m_path, dev);
      auto inputs = std::vector<torch::IValue>();
      inputs.push_back(torch::randn({batch_size_, 3}, dev));
      auto out = m.forward(inputs).toTensor();
      CPPUNIT_ASSERT_EQUAL(dev, out.device());
    });
  }

  void TestModelJIT::testAsyncExecution() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = edm::FileInPath(modelPath).fullPath();

    forEachCudaDevice([&](auto dev) {
      c10::cuda::CUDAGuard guard(dev);
      cudaStream_t stream;
      cudaError_t err = cudaStreamCreate(&stream);
      if (err != cudaSuccess)
        CPPUNIT_FAIL("cudaStreamCreate failed");

      auto default_stream = c10::cuda::getCurrentCUDAStream(dev.index());
      auto torch_stream = c10::cuda::getStreamFromExternal(stream, dev.index());
      c10::cuda::setCurrentCUDAStream(torch_stream);

      auto m = cms::torch::Model(m_path);
      m.to(dev, true);

      auto inputs = std::vector<torch::IValue>();
      inputs.push_back(torch::randn({batch_size_, 3}, dev));

      for (uint32_t i = 0; i < 10; ++i) {
        auto out = m.forward(inputs);
      }

      c10::cuda::setCurrentCUDAStream(default_stream);
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    });
  }

}  // namespace torchtest

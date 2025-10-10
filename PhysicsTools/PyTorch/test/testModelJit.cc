#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "PhysicsTools/PyTorch/interface/Model.h"
#include "PhysicsTools/PyTorch/test/Nvtx.h"
#include "PhysicsTools/PyTorch/test/testTorchBase.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

namespace torchtest {

  using namespace cms::torch;

  class TestModelJIT : public testTorchBase {
  public:
    std::string script() const override;

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

    const int64_t batch_size_ = 2 << 10;
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestModelJIT);

  std::string TestModelJIT::script() const { return "testExportLinearDnn.py"; }

  void TestModelJIT::testCtor_DefaultDeviceIsCpu() {
    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = Model(m_path);

    CPPUNIT_ASSERT_EQUAL(::torch::kCPU, m.device().type());
  }

  void TestModelJIT::testCtor_ExplicitDeviceIsHonored() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto dev = ::torch::Device(::torch::kCUDA, 0);
    auto m = Model(m_path, dev);

    CPPUNIT_ASSERT_EQUAL(dev, m.device());
  }

  void TestModelJIT::testCtor_BadModelPathThrows() {
    auto m_path = modelPath() + "/not_existing_model.pt";
    CPPUNIT_ASSERT_THROW(Model m(m_path), cms::Exception);
  }

  void TestModelJIT::testToDevice_UpdatesUnderlyingState() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = Model(m_path);
    auto dev = ::torch::Device(::torch::kCUDA, 0);
    m.to(dev);

    CPPUNIT_ASSERT_EQUAL(dev, m.device());

    m.to(::torch::kCPU);
    CPPUNIT_ASSERT_EQUAL(::torch::kCPU, m.device().type());
  }

  void TestModelJIT::testToDevice_NonBlocking() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = Model(m_path);
    auto dev = ::torch::Device(::torch::kCUDA, 0);
    Nvtx range("testToDevice_NonBlocking");
    m.to(dev, true);
    range.end();

    CPPUNIT_ASSERT_EQUAL(dev, m.device());
  }

  void TestModelJIT::testForward_IdempotentOutput() {
    auto m_path = modelPath() + "/linear_dnn.pt";
    auto m = Model(m_path);
    auto inputs = std::vector<torch::IValue>();
    inputs.push_back(torch::randn({batch_size_, 3}));
    auto out1 = m.forward(inputs).toTensor();
    auto out2 = m.forward(inputs).toTensor();
    CPPUNIT_ASSERT(out1.equal(out2));
  }

  void TestModelJIT::testForward_OutputOnCorrectDevice() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto dev = ::torch::Device(::torch::kCUDA, 0);
    auto m = Model(m_path, dev);
    auto inputs = std::vector<torch::IValue>();
    inputs.push_back(torch::randn({batch_size_, 3}, dev));
    auto out = m.forward(inputs).toTensor();
    CPPUNIT_ASSERT_EQUAL(dev, out.device());
  }

  void TestModelJIT::testAsyncExecution() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
      CPPUNIT_FAIL("cudaStreamCreate failed");

    auto m_path = modelPath() + "/linear_dnn.pt";
    auto dev = ::torch::Device(::torch::kCUDA, 0);

    // set torch stream from external
    auto default_stream = c10::cuda::getCurrentCUDAStream();
    auto torch_stream = c10::cuda::getStreamFromExternal(stream, dev.index());
    c10::cuda::setCurrentCUDAStream(torch_stream);

    // async model load and inference check
    Nvtx range("testAsyncExecutionModel");
    Nvtx mload("modelLoad");
    auto m = Model(m_path);
    m.to(dev, true);
    mload.end();

    Nvtx inbuf("inputBuffers");
    auto inputs = std::vector<torch::IValue>();
    inputs.push_back(torch::randn({batch_size_, 3}, dev));
    inbuf.end();

    for (uint32_t i = 0; i < 10; ++i) {
      Nvtx iter(("forwardPass:" + std::to_string(i)).c_str());
      auto out = m.forward(inputs);
      iter.end();
    }
    range.end();

    // restore the default stream
    c10::cuda::setCurrentCUDAStream(default_stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }

}  // namespace torchtest
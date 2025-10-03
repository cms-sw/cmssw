#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/Utilities/interface/FileInPath.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "PhysicsTools/PyTorch/interface/ModelAOT.h"
#include "PhysicsTools/PyTorch/test/Nvtx.h"
#include "PhysicsTools/PyTorch/test/testUtilities.h"

namespace torchtest {

  using namespace cms::torch;

  class TestModelAOT : public CppUnit::TestFixture {
  public:
    void testCpu();
    void testCuda();
    void testAsyncExecutionExplicitStream();
    void testAsyncExecutionImplicitStream();

  private:
    CPPUNIT_TEST_SUITE(TestModelAOT);

    CPPUNIT_TEST(testCpu);
    CPPUNIT_TEST(testCuda);
    CPPUNIT_TEST(testAsyncExecutionExplicitStream);
    CPPUNIT_TEST(testAsyncExecutionImplicitStream);

    CPPUNIT_TEST_SUITE_END();

    const int64_t batch_size_ = 2 << 10;
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestModelAOT);

  void TestModelAOT::testCpu() {
    auto m_path = edm::FileInPath("PhysicsTools/PyTorch/models/regression_cpu.pt2").fullPath();
    auto m = ModelAOT(m_path);

    std::vector<::torch::IValue> inputs;
    inputs.push_back(torch::ones({batch_size_, 3}, m.device()));

    std::vector<at::Tensor> inputs_tensor;
    for (const auto& val : inputs)
      inputs_tensor.push_back(val.toTensor());

    auto outputs = m.forward(inputs_tensor);
    for (const auto& val : outputs) {
      CPPUNIT_ASSERT(::torch::allclose(val, ::torch::full_like(val, 0.5f)));
    }
  }

  void TestModelAOT::testCuda() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    auto m_path = edm::FileInPath("PhysicsTools/PyTorch/models/regression_cuda.pt2").fullPath();
    auto m = ModelAOT(m_path);

    std::vector<::torch::IValue> inputs;
    inputs.push_back(torch::ones({batch_size_, 3}, m.device()));

    std::vector<at::Tensor> inputs_tensor;
    for (const auto& val : inputs)
      inputs_tensor.push_back(val.toTensor());

    auto outputs = m.forward(inputs_tensor);
    for (const auto& val : outputs) {
      CPPUNIT_ASSERT(::torch::allclose(val, ::torch::full_like(val, 0.5f)));
    }
  }

  void TestModelAOT::testAsyncExecutionExplicitStream() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    Nvtx test("test");

    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
      CPPUNIT_FAIL("cudaStreamCreate failed");

    auto dev = ::torch::Device(::torch::kCUDA, 0);
    auto m_path = edm::FileInPath("PhysicsTools/PyTorch/models/regression_cpu.pt2").fullPath();

    // async model load and inference check
    Nvtx range("testAsyncExecutionModelExplicitStream");
    Nvtx mload("modelLoad");
    auto m = ModelAOT(m_path);
    mload.end();

    Nvtx inbuf("inputBuffers");
    auto inputs = std::vector<torch::IValue>();
    inputs.push_back(torch::randn({batch_size_, 3}, dev));
    std::vector<at::Tensor> inputs_tensor;
    for (const auto& val : inputs)
      inputs_tensor.push_back(val.toTensor());
    inbuf.end();

    for (uint32_t i = 0; i < 10; ++i) {
      Nvtx iter(("forwardPass:" + std::to_string(i)).c_str());
      auto out = m.forward(inputs_tensor, stream);
      iter.end();
    }
    range.end();

    // restore the default stream
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    test.end();
  }

  void TestModelAOT::testAsyncExecutionImplicitStream() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

    Nvtx test("test");

    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
      CPPUNIT_FAIL("cudaStreamCreate failed");

    auto dev = ::torch::Device(::torch::kCUDA, 0);
    auto m_path = edm::FileInPath("PhysicsTools/PyTorch/models/regression_cuda.pt2").fullPath();

    // set torch stream from external
    auto default_stream = c10::cuda::getCurrentCUDAStream();
    auto torch_stream = c10::cuda::getStreamFromExternal(stream, dev.index());
    c10::cuda::setCurrentCUDAStream(torch_stream);

    // async model load and inference check
    Nvtx range("testAsyncExecutionImplicitStream");
    Nvtx mload("modelLoad");
    auto m = ModelAOT(m_path);
    mload.end();

    Nvtx inbuf("inputBuffers");
    auto inputs = std::vector<torch::IValue>();
    inputs.push_back(torch::randn({batch_size_, 3}, dev));
    std::vector<at::Tensor> inputs_tensor;
    for (const auto& val : inputs)
      inputs_tensor.push_back(val.toTensor());
    inbuf.end();

    for (uint32_t i = 0; i < 10; ++i) {
      Nvtx iter(("forwardPass:" + std::to_string(i)).c_str());
      auto out = m.forward(inputs_tensor, torch_stream);
      iter.end();
    }
    range.end();

    // restore the default stream
    c10::cuda::setCurrentCUDAStream(default_stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    test.end();
  }

}  // namespace torchtest
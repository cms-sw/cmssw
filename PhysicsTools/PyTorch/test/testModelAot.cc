#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/Utilities/interface/FileInPath.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "PhysicsTools/PyTorch/interface/ModelAOT.h"
#include "PhysicsTools/PyTorch/test/testUtilities.h"

namespace torchtest {

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

    const int64_t batch_size_ = 8;
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestModelAOT);

  void TestModelAOT::testCpu() {
    auto m_path = edm::FileInPath("PhysicsTools/PyTorch/models/regression_cpu.pt2").fullPath();
    auto m = cms::torch::ModelAOT(m_path);

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
    auto m = cms::torch::ModelAOT(m_path);

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

    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
      CPPUNIT_FAIL("cudaStreamCreate failed");

    auto dev = ::torch::Device(::torch::kCUDA, 0);
    auto m_path = edm::FileInPath("PhysicsTools/PyTorch/models/regression_cpu.pt2").fullPath();

    // async model load and inference check
    auto m = cms::torch::ModelAOT(m_path);

    auto inputs = std::vector<torch::IValue>();
    inputs.push_back(torch::randn({batch_size_, 3}, dev));
    std::vector<at::Tensor> inputs_tensor;
    for (const auto& val : inputs)
      inputs_tensor.push_back(val.toTensor());

    for (uint32_t i = 0; i < 10; ++i) {
      auto out = m.forward(inputs_tensor, stream);
    }

    // restore the default stream
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }

  void TestModelAOT::testAsyncExecutionImplicitStream() {
    // disable test on non-CUDA devices
    if (!cms::cudatest::testDevices())
      return;

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
    auto m = cms::torch::ModelAOT(m_path);

    auto inputs = std::vector<torch::IValue>();
    inputs.push_back(torch::randn({batch_size_, 3}, dev));
    std::vector<at::Tensor> inputs_tensor;
    for (const auto& val : inputs)
      inputs_tensor.push_back(val.toTensor());

    for (uint32_t i = 0; i < 10; ++i) {
      auto out = m.forward(inputs_tensor, torch_stream);
    }

    // restore the default stream
    c10::cuda::setCurrentCUDAStream(default_stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }

}  // namespace torchtest

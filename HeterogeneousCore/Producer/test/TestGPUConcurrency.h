#ifndef HeterogeneousCore_Producer_test_TestGPUConcurrency_h
#define HeterogeneousCore_Producer_test_TestGPUConcurrency_h

#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"
//#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

class TestGPUConcurrencyAlgo;

/**
 * The purpose of this test is to demonstrate running multiple kernels concurrently on a GPU,
 * associated to different framework streams on he CPU.
 */
class TestGPUConcurrency: public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices <
                                                                       heterogeneous::GPUCuda,
                                                                       heterogeneous::CPU
                                                                       > > {
public:
  explicit TestGPUConcurrency(edm::ParameterSet const& config);
  ~TestGPUConcurrency() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using OutputType = HeterogeneousProductImpl<heterogeneous::CPUProduct<unsigned int>,
                                              heterogeneous::GPUCudaProduct<unsigned int>>;

  void beginStreamGPUCuda(edm::StreamID streamId, cuda::stream_t<>& cudaStream) override;
  void acquireGPUCuda(const edm::HeterogeneousEvent& event, const edm::EventSetup& setup, cuda::stream_t<>& cudaStream) override;
  void produceGPUCuda(edm::HeterogeneousEvent& event, const edm::EventSetup& setup, cuda::stream_t<>& cudaStream) override;
  void produceCPU(edm::HeterogeneousEvent& event, const edm::EventSetup& setup) override;

// GPU code
private:
  TestGPUConcurrencyAlgo * algo_;

// data members
private:
  unsigned int blocks_;
  unsigned int threads_;
  unsigned int sleep_;
};

#endif // HeterogeneousCore_Producer_test_TestGPUConcurrency_h

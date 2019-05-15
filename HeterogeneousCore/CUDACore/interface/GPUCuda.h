#ifndef HeterogeneousCore_CUDAServices_GPUCuda_h
#define HeterogeneousCore_CUDAServices_GPUCuda_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/Producer/interface/DeviceWrapper.h"
#include "HeterogeneousCore/Producer/interface/HeterogeneousEvent.h"

#include <cuda/api_wrappers.h>

#include <memory>

namespace heterogeneous {
  class GPUCuda {
  public:
    using CallbackType = std::function<void(cuda::device::id_t, cuda::stream::id_t, cuda::status_t)>;

    explicit GPUCuda(const edm::ParameterSet& iConfig);
    virtual ~GPUCuda() noexcept(false);

    void call_beginStreamGPUCuda(edm::StreamID id);
    bool call_acquireGPUCuda(DeviceBitSet inputLocation, edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder);
    void call_produceGPUCuda(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup);

    static void fillPSetDescription(edm::ParameterSetDescription& desc);

  private:
    virtual void beginStreamGPUCuda(edm::StreamID id, cuda::stream_t<>& cudaStream) {};
    virtual void acquireGPUCuda(const edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, cuda::stream_t<>& cudaStream) = 0;
    virtual void produceGPUCuda(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, cuda::stream_t<>& cudaStream) = 0;

    std::unique_ptr<cuda::stream_t<>> cudaStream_;
    int deviceId_ = -1; // device assigned to this edm::Stream
    bool enabled_;
    const bool forced_;
  };
  DEFINE_DEVICE_WRAPPER(GPUCuda, HeterogeneousDevice::kGPUCuda);
}

#endif

#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "HeterogeneousHGCalProducerMemoryWrapper.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"

#include "CUDADataFormats/HGCal/interface/HGCRecHitGPUProduct.h"

class HEBRecHitGPUtoSoA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit HEBRecHitGPUtoSoA(const edm::ParameterSet& ps);
  ~HEBRecHitGPUtoSoA() override;

  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  cms::cuda::ContextState ctxState_;
  edm::EDGetTokenT<cms::cuda::Product<HGCRecHitGPUProduct>> recHitGPUToken_;
  edm::EDPutTokenT<HGCRecHitSoA> recHitCPUSoAToken_;

  void allocate_memory_(const uint32_t&, const uint32_t&, const uint32_t&, const cudaStream_t&);

  std::unique_ptr<HGCRecHitSoA> recHitsSoA_;
  HGCRecHitSoA d_calibSoA_;
  std::byte* prodMem_;
};

HEBRecHitGPUtoSoA::HEBRecHitGPUtoSoA(const edm::ParameterSet& ps)
    : recHitGPUToken_{consumes<cms::cuda::Product<HGCRecHitGPUProduct>>(
          ps.getParameter<edm::InputTag>("HEBRecHitGPUTok"))},
      recHitCPUSoAToken_(produces<HGCRecHitSoA>()) {}

HEBRecHitGPUtoSoA::~HEBRecHitGPUtoSoA() {}

void HEBRecHitGPUtoSoA::acquire(edm::Event const& event,
                                edm::EventSetup const& setup,
                                edm::WaitingTaskWithArenaHolder w) {
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(w)};
  const auto& gpuRecHits = ctx.get(event, recHitGPUToken_);
  prodMem_ = gpuRecHits.get();

  recHitsSoA_ = std::make_unique<HGCRecHitSoA>();

  allocate_memory_(gpuRecHits.nHits(), gpuRecHits.stride(), gpuRecHits.nBytes(), ctx.stream());
  KernelManagerHGCalRecHit km(*recHitsSoA_, d_calibSoA_);
  km.transfer_soa_to_host(ctx.stream());
}

void HEBRecHitGPUtoSoA::produce(edm::Event& event, const edm::EventSetup& setup) {
  event.put(std::move(recHitsSoA_), "");
}

void HEBRecHitGPUtoSoA::allocate_memory_(const uint32_t& nhits,
                                         const uint32_t& stride,
                                         const uint32_t& nbytes,
                                         const cudaStream_t& stream) {
  //_allocate memory for calibrated hits on the host
  memory::allocation::calibRecHitHost(nhits, stride, *recHitsSoA_, stream);
  //point SoA to allocated memory for calibrated hits on the device
  memory::allocation::calibRecHitDevice(nhits, stride, nbytes, d_calibSoA_, prodMem_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HEBRecHitGPUtoSoA);

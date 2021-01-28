#include "EERecHitGPUtoSoA.h"

EERecHitGPUtoSoA::EERecHitGPUtoSoA(const edm::ParameterSet& ps)
    : recHitGPUToken_{consumes<cms::cuda::Product<HGCRecHitGPUProduct>>(ps.getParameter<edm::InputTag>("EERecHitGPUTok"))},
      recHitCPUSoAToken_(produces<HGCRecHitSoA>()) {
  d_calibSoA_ = new HGCRecHitSoA();
}

EERecHitGPUtoSoA::~EERecHitGPUtoSoA() { delete d_calibSoA_; }

void EERecHitGPUtoSoA::acquire(edm::Event const& event,
                               edm::EventSetup const& setup,
                               edm::WaitingTaskWithArenaHolder w) {
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(w)};
  const auto& gpuRecHits = ctx.get(event, recHitGPUToken_);
  prodMem_ = gpuRecHits.get();

  recHitsSoA_ = std::make_unique<HGCRecHitSoA>();

  allocate_memory_(gpuRecHits.nHits(), gpuRecHits.stride(), gpuRecHits.nBytes(), ctx.stream());

  KernelManagerHGCalRecHit km(recHitsSoA_.get(), d_calibSoA_);
  km.transfer_soa_to_host(ctx.stream());
}

void EERecHitGPUtoSoA::produce(edm::Event& event, const edm::EventSetup& setup) {
  event.put(std::move(recHitsSoA_), "");
}

void EERecHitGPUtoSoA::allocate_memory_(const uint32_t& nhits,
                                        const uint32_t& stride,
                                        const uint32_t& nbytes,
                                        const cudaStream_t& stream) {
  //_allocate memory for calibrated hits on the host
  memory::allocation::calibRecHitHost(nhits, stride, recHitsSoA_.get(), stream);
  //point SoA to allocated memory for calibrated hits on the device
  memory::allocation::calibRecHitDevice(nhits, stride, nbytes, d_calibSoA_, prodMem_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EERecHitGPUtoSoA);

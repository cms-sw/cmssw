#ifndef RecoParticleFlow_PFRecHitProducer_plugins_alpaka_PFRecHitProducerKernel_h
#define RecoParticleFlow_PFRecHitProducer_plugins_alpaka_PFRecHitProducerKernel_h

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "CalorimeterDefinitions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename CAL>
  class PFRecHitProducerKernel {
  public:
    PFRecHitProducerKernel(Queue& queue, const uint32_t num_recHits);

    // Run kernel: apply filters to rec hits and construct PF rec hits
    // This may be executed multiple times per event
    void processRecHits(Queue& queue,
                        const typename CAL::CaloRecHitSoATypeDevice& recHits,
                        const typename CAL::ParameterType& params,
                        reco::PFRecHitDeviceCollection& pfRecHits);

    // Run kernel: Associate topology information (position, neighbours)
    void associateTopologyInfo(Queue& queue,
                               const typename CAL::TopologyTypeDevice& topology,
                               reco::PFRecHitDeviceCollection& pfRecHits);

  private:
    cms::alpakatools::device_buffer<Device, uint32_t[]> denseId2pfRecHit_;
    cms::alpakatools::device_buffer<Device, uint32_t> num_pfRecHits_;
    WorkDiv<Dim1D> work_div_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoParticleFlow_PFRecHitProducer_plugins_alpaka_PFRecHitProducerKernel_h

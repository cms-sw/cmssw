#ifndef RecoParticleFlow_PFRecHitProducer_PFRecHitProducerKernel_h
#define RecoParticleFlow_PFRecHitProducer_PFRecHitProducerKernel_h

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/CaloRecHitDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PFRecHitProducerKernel {
  public:
    void execute(Queue& queue, const CaloRecHitDeviceCollection& recHits, PFRecHitDeviceCollection& collection) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif 
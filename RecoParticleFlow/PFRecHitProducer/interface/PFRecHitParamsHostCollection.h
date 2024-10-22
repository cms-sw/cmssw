#ifndef RecoParticleFlow_PFRecHitProducer_interface_PFRecHitParamsHostCollection_h
#define RecoParticleFlow_PFRecHitProducer_interface_PFRecHitParamsHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitParamsSoA.h"

namespace reco {
  using PFRecHitHCALParamsHostCollection = PortableHostCollection<PFRecHitHCALParamsSoA>;
  using PFRecHitECALParamsHostCollection = PortableHostCollection<PFRecHitECALParamsSoA>;
}  // namespace reco

#endif  // RecoParticleFlow_PFRecHitProducer_interface_PFRecHitParamsHostCollection_h

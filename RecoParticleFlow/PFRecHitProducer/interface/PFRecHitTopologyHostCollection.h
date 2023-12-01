#ifndef RecoParticleFlow_PFRecHitProducer_interface_PFRecHitTopologyHostCollection_h
#define RecoParticleFlow_PFRecHitProducer_interface_PFRecHitTopologyHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitTopologySoA.h"

namespace reco {
  using PFRecHitHCALTopologyHostCollection = PortableHostCollection<PFRecHitHCALTopologySoA>;
  using PFRecHitECALTopologyHostCollection = PortableHostCollection<PFRecHitECALTopologySoA>;
}  // namespace reco

#endif  // RecoParticleFlow_PFRecHitProducer_interface_PFRecHitTopologyHostCollection_h

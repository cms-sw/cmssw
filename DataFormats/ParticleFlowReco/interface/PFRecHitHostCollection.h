#ifndef DataFormats_ParticleFlowReco_interface_PFRecHitHostCollection_h
#define DataFormats_ParticleFlowReco_interface_PFRecHitHostCollection_h

#include "DataFormats/ParticleFlowReco/interface/PFRecHitSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace reco {

  using PFRecHitHostCollection = PortableHostCollection<PFRecHitSoA>;

}  // namespace reco

#endif  // DataFormats_ParticleFlowReco_interface_PFRecHitHostCollection_h

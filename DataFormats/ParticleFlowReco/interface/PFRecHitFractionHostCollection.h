#ifndef DataFormats_ParticleFlowReco_interface_PFRecHitFractionHostCollection_h
#define DataFormats_ParticleFlowReco_interface_PFRecHitFractionHostCollection_h

#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace reco {
  using PFRecHitFractionHostCollection = PortableHostCollection<PFRecHitFractionSoA>;
}

#endif  // DataFormats_ParticleFlowReco_interface_PFRecHitFractionHostCollection_h

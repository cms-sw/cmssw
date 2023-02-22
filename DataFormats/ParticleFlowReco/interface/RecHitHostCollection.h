#ifndef ParticleFlowReco_RecHitHostCollection_h
#define ParticleFlowReco_RecHitHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/RecHitSoA.h"

namespace portableRecHitSoA {

  using RecHitHostCollection = PortableHostCollection<RecHitSoA>;

}  // namespace portableRecHitSoA

#endif
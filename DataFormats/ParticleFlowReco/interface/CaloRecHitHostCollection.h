#ifndef DataFormats_ParticleFlowReco_interface_CaloRecHitHostCollection_h
#define DataFormats_ParticleFlowReco_interface_CaloRecHitHostCollection_h

#include "DataFormats/ParticleFlowReco/interface/CaloRecHitSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace reco {

  using CaloRecHitHostCollection = PortableHostCollection<CaloRecHitSoA>;

}  // namespace reco

#endif  // DataFormats_ParticleFlowReco_interface_CaloRecHitHostCollection_h

#ifndef DataFormats_ParticleFlowReco_interface_PFClusterHostCollection_h
#define DataFormats_ParticleFlowReco_interface_PFClusterHostCollection_h

#include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace reco {

  using PFClusterHostCollection = PortableHostCollection<PFClusterSoA>;

}  // namespace reco

#endif  // DataFormats_ParticleFlowReco_interface_PFClusterHostCollection_h

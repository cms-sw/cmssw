#ifndef RecoParticleFlow_PFRecHitProducer_interface_PFClusteringEdgeVarsSoA_h
#define RecoParticleFlow_PFRecHitProducer_interface_PFClusteringEdgeVarsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFClusteringEdgeVarsSoALayout,
                      SOA_COLUMN(int, pfrh_edgeIdx),   // needs nRH + 1 allocation
                      SOA_COLUMN(int, pfrh_edgeList))  // needs nRH + maxNeighbors allocation

  using PFClusteringEdgeVarsSoA = PFClusteringEdgeVarsSoALayout<>;
}  // namespace reco

#endif

#ifndef RecoParticleFlow_PFRecHitProducer_interface_PFMultiDepthClusteringEdgeVarsSoA_h
#define RecoParticleFlow_PFRecHitProducer_interface_PFMultiDepthClusteringEdgeVarsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFMultiDepthClusteringEdgeVarsSoALayout,
                      SOA_COLUMN(int, mdpf_adjacencyIndex),  // needs nClusters+1 allocation
                      SOA_COLUMN(int, mdpf_adjacencyList)    // needs 2*nClusters allocation
  )

  using PFMultiDepthClusteringEdgeVarsSoA = PFMultiDepthClusteringEdgeVarsSoALayout<>;
}  // namespace reco

#endif

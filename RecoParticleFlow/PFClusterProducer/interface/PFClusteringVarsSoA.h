#ifndef RecoParticleFlow_PFClusterProducer_interface_PFClusteringVarsSoA_h
#define RecoParticleFlow_PFClusterProducer_interface_PFClusteringVarsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFClusteringVarsSoALayout,
                      SOA_COLUMN(int, pfrh_topoId),
                      SOA_COLUMN(int, pfrh_isSeed),
                      SOA_COLUMN(int, pfrh_passTopoThresh),
                      SOA_COLUMN(int, topoSeedCount),
                      SOA_COLUMN(int, topoRHCount),
                      SOA_COLUMN(int, seedFracOffsets),
                      SOA_COLUMN(int, topoSeedOffsets),
                      SOA_COLUMN(int, topoSeedList),
                      SOA_SCALAR(int, pcrhFracSize),
                      SOA_COLUMN(int, rhCount),
                      SOA_SCALAR(int, nEdges),
                      SOA_COLUMN(int, rhcount),
                      SOA_SCALAR(int, nTopos),
                      SOA_COLUMN(int, topoIds),
                      SOA_SCALAR(int, nRHFracs),
                      SOA_COLUMN(int, rhIdxToSeedIdx))

  using PFClusteringVarsSoA = PFClusteringVarsSoALayout<>;
}  // namespace reco

#endif

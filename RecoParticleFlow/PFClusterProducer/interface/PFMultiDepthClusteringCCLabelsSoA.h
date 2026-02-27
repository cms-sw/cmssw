#ifndef RecoParticleFlow_PFClusterProducer_interface_PFMultiDepthClusteringCCLabelsSoA_h
#define RecoParticleFlow_PFClusterProducer_interface_PFMultiDepthClusteringCCLabelsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFMultiDepthClusteringCCLabelsSoALayout,
                      SOA_COLUMN(int, mdpf_topoId),
                      SOA_COLUMN(int, workl),
                      SOA_SCALAR(int, topH),
                      SOA_SCALAR(int, posH),
                      SOA_SCALAR(int, topL),
                      SOA_SCALAR(int, posL),
                      SOA_SCALAR(int, size))

  using PFMultiDepthClusteringCCLabelsSoA = PFMultiDepthClusteringCCLabelsSoALayout<>;
}  // namespace reco

#endif

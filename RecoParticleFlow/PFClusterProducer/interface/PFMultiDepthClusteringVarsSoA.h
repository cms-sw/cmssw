#ifndef RecoParticleFlow_PFClusterProducer_interface_PFMultiDepthClusteringVarsSoA_h
#define RecoParticleFlow_PFClusterProducer_interface_PFMultiDepthClusteringVarsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFMultiDepthClusteringVarsSoALayout,
                      SOA_COLUMN(float, depth),
                      SOA_COLUMN(double, etaRMS2),
                      SOA_COLUMN(double, phiRMS2),
                      SOA_COLUMN(float, energy),
                      SOA_COLUMN(float, eta),
                      SOA_COLUMN(float, phi),
                      SOA_SCALAR(int, size))

  using PFMultiDepthClusteringVarsSoA = PFMultiDepthClusteringVarsSoALayout<>;
}  // namespace reco

#endif

#ifndef RecoParticleFlow_PFClusterProducer_interface_PFMultiDepthClusteringVarsSoA_h
#define RecoParticleFlow_PFClusterProducer_interface_PFMultiDepthClusteringVarsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFMultiDepthClusteringVarsSoALayout,
                      SOA_COLUMN(float, depth),
                      //SOA_COLUMN(int, seedRHIdx),
                      //SOA_COLUMN(int, rhfracSize),
                      //SOA_COLUMN(int, rhfracOffset),
                      SOA_COLUMN(double, etaRMS2),
                      SOA_COLUMN(double, phiRMS2),
                      SOA_COLUMN(float, energy),
                      SOA_COLUMN(float, eta),
                      SOA_COLUMN(float, phi),
                      SOA_COLUMN(int, mdpf_topoId),
                      //SOA_COLUMN(int, mdpf_component), //list of component contents (always start with component root id)
                      //SOA_COLUMN(int, mdpf_componentIndex),
                      //SOA_COLUMN(int, mdpf_componentEnergy),
                      //SOA_SCALAR(int, mdpf_nTopos),                      
                      SOA_SCALAR(int, size)
                    )
                    
  using PFMultiDepthClusteringVarsSoA = PFMultiDepthClusteringVarsSoALayout<>;
}  // namespace reco

#endif

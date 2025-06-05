#ifndef DataFormats_ParticleFlowReco_interface_PFClusterSoA_h
#define DataFormats_ParticleFlowReco_interface_PFClusterSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFClusterSoALayout,
                      SOA_COLUMN(int, depth),
                      SOA_COLUMN(int, seedRHIdx),
                      SOA_COLUMN(int, topoId),
                      SOA_COLUMN(int, rhfracSize),
                      SOA_COLUMN(int, rhfracOffset),
                      SOA_COLUMN(float, energy),
                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      SOA_COLUMN(float, z),
                      SOA_COLUMN(int, topoRHCount),
                      SOA_SCALAR(int, nTopos),
                      SOA_SCALAR(int, nSeeds),
                      SOA_SCALAR(int, nRHFracs),
                      SOA_SCALAR(int, size)  // nRH
  )
  using PFClusterSoA = PFClusterSoALayout<>;
}  // namespace reco

#endif  // DataFormats_ParticleFlowReco_interface_PFClusterSoA_h

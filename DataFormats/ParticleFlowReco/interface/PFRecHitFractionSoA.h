#ifndef DataFormats_ParticleFlowReco_interface_PFRecHitFractionSoA_h
#define DataFormats_ParticleFlowReco_interface_PFRecHitFractionSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFRecHitFractionSoALayout,
                      SOA_COLUMN(float, frac),
                      SOA_COLUMN(int, pfrhIdx),
                      SOA_COLUMN(int, pfcIdx))

  using PFRecHitFractionSoA = PFRecHitFractionSoALayout<>;
}  // namespace reco

#endif  // DataFormats_ParticleFlowReco_interface_PFRecHitFractionSoA_h

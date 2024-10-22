#ifndef DataFormats_ParticleFlowReco_interface_CaloRecHitSoA_h
#define DataFormats_ParticleFlowReco_interface_CaloRecHitSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  GENERATE_SOA_LAYOUT(CaloRecHitSoALayout,
                      SOA_COLUMN(uint32_t, detId),
                      SOA_COLUMN(float, energy),
                      SOA_COLUMN(float, time),
                      SOA_COLUMN(uint32_t, flags))

  using CaloRecHitSoA = CaloRecHitSoALayout<>;

}  // namespace reco

#endif  // DataFormats_ParticleFlowReco_interface_CaloRecHitSoA_h

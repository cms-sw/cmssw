#ifndef ParticleFlowReco_CaloRecHitSoA_h
#define ParticleFlowReco_CaloRecHitSoA_h

#include <cstdint>
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

GENERATE_SOA_LAYOUT(CaloRecHitSoALayout,
  SOA_COLUMN(uint32_t, detId),
  SOA_COLUMN(float, energy),
  SOA_COLUMN(float, time)
)

using CaloRecHitSoA = CaloRecHitSoALayout<>;

#endif

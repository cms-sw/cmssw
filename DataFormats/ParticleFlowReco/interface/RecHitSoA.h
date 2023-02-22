#ifndef ParticleFlowReco_RecHitSoA_h
#define ParticleFlowReco_RecHitSoA_h

#include <cstdint>
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace portableRecHitSoA {

  GENERATE_SOA_LAYOUT(RecHitSoALayout,
                      SOA_COLUMN(uint32_t, detId),
                      SOA_COLUMN(float, energy),
                      SOA_COLUMN(float, time)
  )

  using RecHitSoA = RecHitSoALayout<>;

}  // namespace portableRecHitSoA 

#endif

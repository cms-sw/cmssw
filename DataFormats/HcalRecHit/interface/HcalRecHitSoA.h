#ifndef DataFormats_HcalRecHit_HcalRecHitSoA_h
#define DataFormats_HcalRecHit_HcalRecHitSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace hcal {

  GENERATE_SOA_LAYOUT(HcalRecHitSoALayout,
                      SOA_COLUMN(uint32_t, detId),
                      SOA_COLUMN(float, energy),
                      SOA_COLUMN(float, chi2),
                      SOA_COLUMN(float, energyM0),
                      SOA_COLUMN(float, timeM0))

  using HcalRecHitSoA = HcalRecHitSoALayout<>;
}  // namespace hcal

#endif

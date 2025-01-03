#ifndef DataFormats_EcalRecHit_EcalRecHitSoA_h
#define DataFormats_EcalRecHit_EcalRecHitSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(EcalRecHitSoALayout,
                    SOA_SCALAR(uint32_t, size),
                    SOA_COLUMN(uint32_t, id),
                    SOA_COLUMN(float, energy),
                    SOA_COLUMN(float, time),
                    SOA_COLUMN(uint32_t, flagBits),
                    SOA_COLUMN(uint32_t, extra))

using EcalRecHitSoA = EcalRecHitSoALayout<>;

#endif

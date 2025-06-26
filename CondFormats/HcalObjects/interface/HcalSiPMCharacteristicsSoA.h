#ifndef CondFormats_HcalObjects_HcalSiPMCharacteristicsSoA_h
#define CondFormats_HcalObjects_HcalSiPMCharacteristicsSoA_h

#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristics.h"

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace hcal {
  GENERATE_SOA_LAYOUT(HcalSiPMCharacteristicsSoALayout,
                      SOA_COLUMN(HcalSiPMCharacteristics::PrecisionItem, precisionItem))
  using HcalSiPMCharacteristicsSoA = HcalSiPMCharacteristicsSoALayout<>;
}  // namespace hcal
#endif

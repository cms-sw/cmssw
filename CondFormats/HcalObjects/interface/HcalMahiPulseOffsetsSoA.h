#ifndef CondFormats_HcalObjects_HcalMahiPulseOffsetsSoA_h
#define CondFormats_HcalObjects_HcalMahiPulseOffsetsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace hcal {
  GENERATE_SOA_LAYOUT(HcalMahiPulseOffsetsSoALayout, SOA_COLUMN(int, offsets))
  using HcalMahiPulseOffsetsSoA = HcalMahiPulseOffsetsSoALayout<>;

}  // namespace hcal
#endif

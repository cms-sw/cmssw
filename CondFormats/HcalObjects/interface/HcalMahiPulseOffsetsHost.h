#ifndef CondFormats_HcalObjects_interface_HcalMahiPulseOffsetsPortable_h
#define CondFormats_HcalObjects_interface_HcalMahiPulseOffsetsPortable_h

#include "CondFormats/HcalObjects/interface/HcalMahiPulseOffsetsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace hcal {
  using HcalMahiPulseOffsetsPortableHost = PortableHostCollection<HcalMahiPulseOffsetsSoA>;
}
#endif

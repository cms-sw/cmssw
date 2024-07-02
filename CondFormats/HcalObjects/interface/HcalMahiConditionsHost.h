#ifndef CondFormats_HcalObjects_interface_HcalMahiConditionsHost_h
#define CondFormats_HcalObjects_interface_HcalMahiConditionsHost_h

#include "CondFormats/HcalObjects/interface/HcalMahiConditionsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace hcal {
  using HcalMahiConditionsPortableHost = PortableHostCollection<HcalMahiConditionsSoA>;
}

#endif

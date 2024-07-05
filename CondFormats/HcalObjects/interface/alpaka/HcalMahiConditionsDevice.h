#ifndef CondFormats_HcalObjects_interface_alpaka_HcalMahiConditionsDevice_h
#define CondFormats_HcalObjects_interface_alpaka_HcalMahiConditionsDevice_h

#include "CondFormats/HcalObjects/interface/HcalMahiConditionsHost.h"
#include "CondFormats/HcalObjects/interface/HcalMahiConditionsSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hcal {

    using ::hcal::HcalMahiConditionsPortableHost;
    using HcalMahiConditionsPortableDevice = PortableCollection<::hcal::HcalMahiConditionsSoA>;
  }  // namespace hcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

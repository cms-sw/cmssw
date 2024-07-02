#ifndef CondFormats_HcalObjects_interface_alpaka_HcalMahiPulseOffsetsPortable_h
#define CondFormats_HcalObjects_interface_alpaka_HcalMahiPulseOffsetsPortable_h

#include "CondFormats/HcalObjects/interface/HcalMahiPulseOffsetsHost.h"
#include "CondFormats/HcalObjects/interface/HcalMahiPulseOffsetsSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hcal {
    using ::hcal::HcalMahiPulseOffsetsPortableHost;
    using HcalMahiPulseOffsetsPortableDevice = PortableCollection<::hcal::HcalMahiPulseOffsetsSoA>;

  }  // namespace hcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

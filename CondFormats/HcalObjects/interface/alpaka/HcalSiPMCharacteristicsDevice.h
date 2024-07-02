#ifndef CondFormats_HcalObjects_interface_alpaka_HcalSiPMCharacteristicsPortable_h
#define CondFormats_HcalObjects_interface_alpaka_HcalSiPMCharacteristicsPortable_h

#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristicsHost.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristicsSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hcal {
    using ::hcal::HcalSiPMCharacteristicsPortableHost;
    using HcalSiPMCharacteristicsPortableDevice = PortableCollection<::hcal::HcalSiPMCharacteristicsSoA>;
  }  // namespace hcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

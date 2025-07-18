#ifndef CondFormats_HcalObjects_interface_HcalSiPMCharacteristicsPortable_h
#define CondFormats_HcalObjects_interface_HcalSiPMCharacteristicsPortable_h

#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristicsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace hcal {
  using HcalSiPMCharacteristicsPortableHost = PortableHostCollection<HcalSiPMCharacteristicsSoA>;
}
#endif

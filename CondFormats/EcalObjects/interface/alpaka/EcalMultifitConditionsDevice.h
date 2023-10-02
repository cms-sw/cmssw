#ifndef CondFormats_EcalObjects_interface_alpaka_EcalMultifitConditionsDevice_h
#define CondFormats_EcalObjects_interface_alpaka_EcalMultifitConditionsDevice_h

#include "CondFormats/EcalObjects/interface/EcalMultifitConditionsHost.h"
#include "CondFormats/EcalObjects/interface/EcalMultifitConditionsSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ::EcalMultifitConditionsHost;
  using EcalMultifitConditionsDevice = PortableCollection<EcalMultifitConditionsSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

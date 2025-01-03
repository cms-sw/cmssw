#ifndef CondFormats_EcalObjects_interface_alpaka_EcalRecHitConditionsDevice_h
#define CondFormats_EcalObjects_interface_alpaka_EcalRecHitConditionsDevice_h

#include "CondFormats/EcalObjects/interface/EcalRecHitConditionsHost.h"
#include "CondFormats/EcalObjects/interface/EcalRecHitConditionsSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ::EcalRecHitConditionsHost;
  using EcalRecHitConditionsDevice = PortableCollection<EcalRecHitConditionsSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

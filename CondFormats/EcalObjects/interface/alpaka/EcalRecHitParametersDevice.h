#ifndef CondFormats_EcalObjects_interface_alpaka_EcalRecHitParametersDevice_h
#define CondFormats_EcalObjects_interface_alpaka_EcalRecHitParametersDevice_h

#include "CondFormats/EcalObjects/interface/EcalRecHitParametersHost.h"
#include "CondFormats/EcalObjects/interface/EcalRecHitParametersSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ::EcalRecHitParametersHost;
  using EcalRecHitParametersDevice = PortableCollection<EcalRecHitParametersSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

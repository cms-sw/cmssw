#ifndef CondFormats_EcalObjects_interface_alpaka_EcalMultifitParametersDevice_h
#define CondFormats_EcalObjects_interface_alpaka_EcalMultifitParametersDevice_h

#include "CondFormats/EcalObjects/interface/EcalMultifitParametersHost.h"
#include "CondFormats/EcalObjects/interface/EcalMultifitParametersSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ::EcalMultifitParametersHost;
  using EcalMultifitParametersDevice = PortableCollection<EcalMultifitParametersSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

#ifndef CondFormats_EcalObjects_interface_alpaka_EcalRecHitParametersDevice_h
#define CondFormats_EcalObjects_interface_alpaka_EcalRecHitParametersDevice_h

#include "CondFormats/EcalObjects/interface/EcalRecHitParameters.h"
#include "CondFormats/EcalObjects/interface/EcalRecHitParametersHost.h"
#include "DataFormats/Portable/interface/alpaka/PortableObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ::EcalRecHitParametersHost;
  using EcalRecHitParametersDevice = PortableObject<EcalRecHitParameters>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(EcalRecHitParametersDevice, EcalRecHitParametersHost);

#endif

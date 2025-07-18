#ifndef RecoLocalCalo_EcalRecProducers_plugins_alpaka_KernelHelpers_h
#define RecoLocalCalo_EcalRecProducers_plugins_alpaka_KernelHelpers_h

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::reconstruction {

  ALPAKA_FN_ACC uint32_t hashedIndexEB(uint32_t id);

  ALPAKA_FN_ACC uint32_t hashedIndexEE(uint32_t id);

  ALPAKA_FN_ACC int32_t laserMonitoringRegionEB(uint32_t id);

  ALPAKA_FN_ACC int32_t laserMonitoringRegionEE(uint32_t id);

  ALPAKA_FN_ACC int32_t rechitSetMasked(uint32_t value, uint32_t x, uint32_t offset, uint32_t width);

  ALPAKA_FN_ACC int32_t rechitGetPower10(float e);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::reconstruction

#endif  // RecoLocalCalo_EcalRecProducers_plugins_alpaka_KernelHelpers_h

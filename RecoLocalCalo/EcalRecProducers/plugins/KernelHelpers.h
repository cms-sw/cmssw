#ifndef RecoLocalCalo_EcalRecProducers_plugins_KernelHelpers_h
#define RecoLocalCalo_EcalRecProducers_plugins_KernelHelpers_h

#include "DataFormats/CaloRecHit/interface/MultifitComputations.h"

#include <cmath>
#include <limits>
#include <type_traits>

#include <Eigen/Dense>

namespace ecal {
  namespace reconstruction {

    __device__ uint32_t hashedIndexEB(uint32_t id);

    __device__ uint32_t hashedIndexEE(uint32_t id);

    __device__ int laser_monitoring_region_EB(uint32_t id);

    __device__ int laser_monitoring_region_EE(uint32_t id);

  }  // namespace reconstruction
}  // namespace ecal

#endif  // RecoLocalCalo_EcalRecProducers_plugins_KernelHelpers_h

#include "HeterogeneousCore/CUDAServices/interface/numberOfDevices.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

namespace cms::cuda {
  int numberOfDevices() {
    edm::Service<CUDAService> cs;
    return cs->enabled() ? cs->numberOfDevices() : 0;
  }
}  // namespace cms::cuda

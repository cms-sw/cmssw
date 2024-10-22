#include "HeterogeneousCore/CUDAServices/interface/numberOfDevices.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAInterface.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

namespace cms::cuda {
  int numberOfDevices() {
    edm::Service<CUDAInterface> cuda;
    return (cuda and cuda->enabled()) ? cuda->numberOfDevices() : 0;
  }
}  // namespace cms::cuda

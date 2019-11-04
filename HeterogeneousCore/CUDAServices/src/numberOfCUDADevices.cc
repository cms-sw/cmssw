#include "HeterogeneousCore/CUDAServices/interface/numberOfCUDADevices.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

int numberOfCUDADevices() {
  edm::Service<CUDAService> cs;
  return cs->enabled() ? cs->numberOfDevices() : 0;
}

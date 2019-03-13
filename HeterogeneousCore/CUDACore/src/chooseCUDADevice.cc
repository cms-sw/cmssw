#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "chooseCUDADevice.h"

namespace cudacore {
  int chooseCUDADevice(edm::StreamID id) {
    edm::Service<CUDAService> cudaService;

    // For startes we "statically" assign the device based on
    // edm::Stream number. This is suboptimal if the number of
    // edm::Streams is not a multiple of the number of CUDA devices
    // (and even then there is no load balancing).
    //
    // TODO: improve the "assignment" logic
    return id % cudaService->numberOfDevices();
  }
}

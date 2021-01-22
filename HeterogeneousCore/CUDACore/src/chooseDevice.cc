#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "chooseDevice.h"

namespace cms::cuda {
  int chooseDevice(edm::StreamID id) {
    edm::Service<CUDAService> cudaService;
    if (not cudaService->enabled()) {
      cms::Exception ex("CUDAError");
      ex << "Unable to choose current device because CUDAService is disabled. If CUDAService was not explicitly\n"
            "disabled in the configuration, the probable cause is that there is no GPU or there is some problem\n"
            "in the CUDA runtime or drivers.";
      ex.addContext("Calling cms::cuda::chooseDevice()");
      throw ex;
    }

    // For startes we "statically" assign the device based on
    // edm::Stream number. This is suboptimal if the number of
    // edm::Streams is not a multiple of the number of CUDA devices
    // (and even then there is no load balancing).
    //
    // TODO: improve the "assignment" logic
    return id % cudaService->numberOfDevices();
  }
}  // namespace cms::cuda

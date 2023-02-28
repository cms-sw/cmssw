#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAInterface.h"

#include "chooseDevice.h"

namespace cms::cuda {
  int chooseDevice(edm::StreamID id) {
    edm::Service<CUDAInterface> cuda;
    if (not cuda or not cuda->enabled()) {
      cms::Exception ex("CUDAError");
      ex << "Unable to choose current device because CUDAService is not preset or disabled.\n"
         << "If CUDAService was not explicitly disabled in the configuration, the probable\n"
         << "cause is that there is no GPU or there is some problem in the CUDA runtime or\n"
         << "drivers.";
      ex.addContext("Calling cms::cuda::chooseDevice()");
      throw ex;
    }

    // For startes we "statically" assign the device based on
    // edm::Stream number. This is suboptimal if the number of
    // edm::Streams is not a multiple of the number of CUDA devices
    // (and even then there is no load balancing).
    //
    // TODO: improve the "assignment" logic
    return id % cuda->numberOfDevices();
  }
}  // namespace cms::cuda

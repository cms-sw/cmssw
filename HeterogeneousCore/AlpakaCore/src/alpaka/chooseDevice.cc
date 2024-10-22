#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/chooseDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaServices/interface/alpaka/AlpakaService.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::detail {
  Device const& chooseDevice(edm::StreamID id) {
    // The idea of checking the AlpakaService status here is that
    // regardless of process.options.accelerators setting the
    // configuration may contain an EDModule for a specific Alpaka
    // backend. Checking the AlpakaService status here ensures that an
    // explanatory error messages even in that case. The information
    // on the intended accelerators is (eventually) communicated from
    // process.options.accelerators to AlpakaServices via
    // ProcessAcceleratorAlpaka.
    edm::Service<ALPAKA_ACCELERATOR_NAMESPACE::AlpakaService> alpakaService;
    if (not alpakaService->enabled()) {
      cms::Exception ex("AlpakaError");
      ex << "Unable to choose current device because AlpakaService is disabled. If AlpakaService was not explicitly\n"
            "disabled in the configuration, the probable cause is that there is no GPU or there is some problem\n"
            "in the platform runtime or drivers.";
      ex.addContext("Calling " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "::detail::chooseDevice()");
      throw ex;
    };

    // For startes we "statically" assign the device based on
    // edm::Stream number. This is suboptimal if the number of
    // edm::Streams is not a multiple of the number of devices
    // (and even then there is no load balancing).

    // TODO: improve the "assignment" logic
    auto const& devices = cms::alpakatools::devices<Platform>();
    return devices[id % devices.size()];
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::detail

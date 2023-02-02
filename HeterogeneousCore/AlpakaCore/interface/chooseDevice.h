#ifndef HeterogeneousCore_AlpakaCore_interface_chooseDevice_h
#define HeterogeneousCore_AlpakaCore_interface_chooseDevice_h

#include <alpaka/alpaka.hpp>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaServices/interface/alpaka/AlpakaService.h"

namespace cms::alpakatools {

  template <typename TPlatform, typename = std::enable_if_t<alpaka::isPlatform<TPlatform>>>
  alpaka::Dev<TPlatform> const& chooseDevice(edm::StreamID id) {
    edm::Service<ALPAKA_TYPE_ALIAS(AlpakaService)> service;
    if (not service->enabled()) {
      cms::Exception ex("RuntimeError");
      ex << "Unable to choose current device because " << ALPAKA_TYPE_ALIAS_NAME(AlpakaService) << " is disabled.\n"
         << "If " << ALPAKA_TYPE_ALIAS_NAME(AlpakaService) << " was not explicitly disabled in the configuration,\n"
         << "the probable cause is that there is no accelerator or there is some problem\n"
         << "with the accelerator runtime or drivers.";
      ex.addContext("Calling cms::alpakatools::chooseDevice()");
      throw ex;
    }

    // For startes we "statically" assign the device based on
    // edm::Stream number. This is suboptimal if the number of
    // edm::Streams is not a multiple of the number of devices
    // (and even then there is no load balancing).

    // TODO: improve the "assignment" logic
    auto const& devices = cms::alpakatools::devices<TPlatform>();
    return devices[id % devices.size()];
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaCore_interface_chooseDevice_h

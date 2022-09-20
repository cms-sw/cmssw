#ifndef HeterogeneousCore_AlpakaCore_interface_chooseDevice_h
#define HeterogeneousCore_AlpakaCore_interface_chooseDevice_h

#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

namespace cms::alpakatools {

  template <typename TPlatform, typename = std::enable_if_t<is_platform_v<TPlatform>>>
  alpaka::Dev<TPlatform> const& chooseDevice(edm::StreamID id) {
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

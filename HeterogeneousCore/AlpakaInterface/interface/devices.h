#ifndef HeterogeneousCore_AlpakaInterface_interface_devices_h
#define HeterogeneousCore_AlpakaInterface_interface_devices_h

#include <vector>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace cms::alpakatools {

  // returns the alpaka accelerator platform
  template <typename TPlatform, typename = std::enable_if_t<alpaka::isPlatform<TPlatform>>>
  inline TPlatform const& platform() {
    // initialise the platform the first time that this function is called
    static const auto platform = TPlatform{};
    return platform;
  }

  // return the alpaka accelerator devices for the given platform
  template <typename TPlatform, typename = std::enable_if_t<alpaka::isPlatform<TPlatform>>>
  inline std::vector<alpaka::Dev<TPlatform>> const& devices() {
    // enumerate all devices the first time that this function is called
    static const auto devices = alpaka::getDevs(platform<TPlatform>());
    return devices;
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_devices_h

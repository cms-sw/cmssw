#ifndef HeterogeneousCore_AlpakaInterface_interface_devices_h
#define HeterogeneousCore_AlpakaInterface_interface_devices_h

#include <cassert>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace cms::alpakatools {

  namespace detail {

    template <typename TPlatform, typename = std::enable_if_t<alpaka::isPlatform<TPlatform>>>
    inline std::vector<alpaka::Dev<TPlatform>> enumerate_devices() {
      using Platform = TPlatform;
      using Device = alpaka::Dev<Platform>;

      std::vector<Device> devices;
      uint32_t n = alpaka::getDevCount<Platform>();
      devices.reserve(n);
      for (uint32_t i = 0; i < n; ++i) {
        devices.push_back(alpaka::getDevByIdx<Platform>(i));
        assert(alpaka::getNativeHandle(devices.back()) == static_cast<int>(i));
      }

      return devices;
    }

  }  // namespace detail

  // return the alpaka accelerator devices for the given platform
  template <typename TPlatform, typename = std::enable_if_t<alpaka::isPlatform<TPlatform>>>
  inline std::vector<alpaka::Dev<TPlatform>> const& devices() {
    static const auto devices = detail::enumerate_devices<TPlatform>();
    return devices;
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_devices_h

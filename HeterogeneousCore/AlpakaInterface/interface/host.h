#ifndef HeterogeneousCore_AlpakaInterface_interface_host_h
#define HeterogeneousCore_AlpakaInterface_interface_host_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"

namespace cms::alpakatools {

  // returns the alpaka host platform
  inline alpaka::PlatformCpu const& host_platform() { return platform<alpaka::PlatformCpu>(); }

  // returns the alpaka host device
  inline alpaka::DevCpu const& host() { return devices<alpaka::PlatformCpu>()[0]; }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_host_h

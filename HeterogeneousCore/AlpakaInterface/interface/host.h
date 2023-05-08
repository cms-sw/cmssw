#ifndef HeterogeneousCore_AlpakaInterface_interface_host_h
#define HeterogeneousCore_AlpakaInterface_interface_host_h

#include <cassert>

#include <alpaka/alpaka.hpp>

namespace cms::alpakatools {

  namespace detail {

    inline alpaka::DevCpu enumerate_host() {
      using Platform = alpaka::PltfCpu;
      using Host = alpaka::DevCpu;

      assert(alpaka::getDevCount<Platform>() == 1);
      Host host = alpaka::getDevByIdx<Platform>(0);
      assert(alpaka::getNativeHandle(host) == 0);

      return host;
    }

  }  // namespace detail

  // returns the alpaka host device
  inline alpaka::DevCpu const& host() {
    static const auto host = detail::enumerate_host();
    return host;
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_host_h

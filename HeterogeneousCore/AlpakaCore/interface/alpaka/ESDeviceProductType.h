#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_ESDeviceProductType_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_ESDeviceProductType_h

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESDeviceProduct.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include <type_traits>

namespace ALPAKA_ACCELERATOR_NAMESPACE::detail {
  // host backends can use TProduct directly
  // all device backends need to be wrapped
  inline constexpr bool useESProductDirectly = std::is_same_v<Platform, alpaka::PlatformCpu>;

  /**
   * Type alias for the actual product type put in an
   * EventSetup record
   */
  template <typename TProduct>
  using ESDeviceProductType = std::conditional_t<useESProductDirectly, TProduct, ESDeviceProduct<TProduct>>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::detail

#endif

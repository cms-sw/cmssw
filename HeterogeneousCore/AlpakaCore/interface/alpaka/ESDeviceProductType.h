#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_ESDeviceProductType_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_ESDeviceProductType_h

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESDeviceProduct.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include <type_traits>

namespace ALPAKA_ACCELERATOR_NAMESPACE::detail {
  /**
   * This "trait" class abstracts the actual product type put in an
   * EventSetup record
   */
  template <typename TProduct>
  struct ESDeviceProductType {
    using type = std::conditional_t<std::is_same_v<Platform, alpaka::PltfCpu>,
                                    // host backends can use TProduct directly
                                    TProduct,
                                    // all device backends need to be wrapped
                                    ESDeviceProduct<TProduct>>;
  };

  template <typename TProduct>
  inline constexpr bool useESProductDirectly = std::is_same_v<typename ESDeviceProductType<TProduct>::type, TProduct>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::detail

#endif

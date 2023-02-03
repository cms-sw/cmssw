#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_DeviceProductType_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_DeviceProductType_h

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::detail {
  /**
   * This "trait" class abstracts the actual product type put in the
   * edm::Event.
   */
  template <typename TProduct>
  struct DeviceProductType {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    // host synchronous backends can use TProduct directly
    using type = TProduct;
#else
    // all device and asynchronous backends need to be wrapped
    using type = edm::DeviceProduct<TProduct>;
#endif
  };

  template <typename TProduct>
  inline constexpr bool useProductDirectly = std::is_same_v<typename DeviceProductType<TProduct>::type, TProduct>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::detail

#endif

#ifndef DataFormats_Portable_interface_PortableProduct_h
#define DataFormats_Portable_interface_PortableProduct_h

#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

namespace traits {

  // trait for a generic SoA-based product
  template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  class PortableProductTrait;

}  // namespace traits

// type alias for a generic SoA-based product
template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
using PortableProduct = typename traits::PortableProductTrait<T, TDev>::ProductType;

#endif  // DataFormats_Portable_interface_PortableProduct_h

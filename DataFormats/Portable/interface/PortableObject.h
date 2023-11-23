#ifndef DataFormats_Portable_interface_PortableObject_h
#define DataFormats_Portable_interface_PortableObject_h

#include <type_traits>

#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

namespace traits {

  // trait for a generic SoA-based product
  template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  class PortableObjectTrait;

}  // namespace traits

// type alias for a generic SoA-based product
template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
using PortableObject = typename traits::PortableObjectTrait<T, TDev>::ProductType;

#endif  // DataFormats_Portable_interface_PortableObject_h

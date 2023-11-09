#ifndef DataFormats_Portable_interface_PortableObject_h
#define DataFormats_Portable_interface_PortableObject_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

namespace traits {

  // trait for a generic SoA-based product
  template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  struct PortableObjectTrait;

}  // namespace traits

// type alias for a generic SoA-based product
template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
using PortableObject = typename traits::PortableObjectTrait<T, TDev>::ProductType;

#endif  // DataFormats_Portable_interface_PortableObject_h

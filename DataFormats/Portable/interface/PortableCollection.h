#ifndef DataFormats_Portable_interface_PortableCollection_h
#define DataFormats_Portable_interface_PortableCollection_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

namespace traits {

  // trait for a generic SoA-based product
  template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  struct PortableCollectionTrait;

}  // namespace traits

// type alias for a generic SoA-based product
template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
using PortableCollection = typename traits::PortableCollectionTrait<T, TDev>::CollectionType;

#endif  // DataFormats_Portable_interface_PortableCollection_h

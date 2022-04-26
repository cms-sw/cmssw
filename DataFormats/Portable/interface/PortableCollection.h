#ifndef DataFormats_Portable_interface_PortableCollection_h
#define DataFormats_Portable_interface_PortableCollection_h

namespace traits {

  // trait for a generic SoA-based product
  template <typename T, typename TDev>
  class PortableCollectionTrait;

}  // namespace traits

// type alias for a generic SoA-based product
template <typename T, typename TDev>
using PortableCollection = typename traits::PortableCollectionTrait<T, TDev>::CollectionType;

#endif  // DataFormats_Portable_interface_PortableCollection_h

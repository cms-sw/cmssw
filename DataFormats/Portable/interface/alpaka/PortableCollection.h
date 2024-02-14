#ifndef DataFormats_Portable_interface_alpaka_PortableCollection_h
#define DataFormats_Portable_interface_alpaka_PortableCollection_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// This header is not used by PortableCollection, but is included here to automatically
// provide its content to users of ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection.
#include "HeterogeneousCore/AlpakaInterface/interface/AssertDeviceMatchesHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // generic SoA-based product in the device (that may be host) memory
  template <typename T>
  using PortableCollection = ::PortableCollection<T, Device>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace ALPAKA_ACCELERATOR_NAMESPACE {

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // Singleton case does not need to be aliased. A special template covers it.

  // This aliasing is needed to work with ROOT serialization. Bare templates make dictionary compilation fail.
  template <typename T0, typename T1>
  using PortableCollection2 = ::PortableHostMultiCollection<T0, T1>;

  template <typename T0, typename T1, typename T2>
  using PortableCollection3 = ::PortableHostMultiCollection<T0, T1, T2>;

  template <typename T0, typename T1, typename T2, typename T3>
  using PortableCollection4 = ::PortableHostMultiCollection<T0, T1, T2, T3>;

  template <typename T0, typename T1, typename T2, typename T3, typename T4>
  using PortableCollection5 = ::PortableHostMultiCollection<T0, T1, T2, T3, T4>;
#else
  // Singleton case does not need to be aliased. A special template covers it.

  // This aliasing is needed to work with ROOT serialization. Bare templates make dictionary compilation fail.
  template <typename T0, typename T1>
  using PortableCollection2 = ::PortableDeviceMultiCollection<Device, T0, T1>;

  template <typename T0, typename T1, typename T2>
  using PortableCollection3 = ::PortableDeviceMultiCollection<Device, T0, T1, T2>;

  template <typename T0, typename T1, typename T2, typename T3>
  using PortableCollection4 = ::PortableDeviceMultiCollection<Device, T0, T1, T2, T3>;

  template <typename T0, typename T1, typename T2, typename T3, typename T4>
  using PortableCollection5 = ::PortableDeviceMultiCollection<Device, T0, T1, T2, T3, T4>;
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace traits {
// specialise the trait for the device provided by the ALPAKA_ACCELERATOR_NAMESPACE
#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  template <typename T0, typename... Args>
  class PortableMultiCollectionTrait<ALPAKA_ACCELERATOR_NAMESPACE::Device, T0, Args...> {
    using CollectionType = ::PortableHostMultiCollection<T0, Args...>;
  };
#else
  template <typename T0, typename... Args>
  class PortableMultiCollectionTrait<ALPAKA_ACCELERATOR_NAMESPACE::Device, T0, Args...> {
    using CollectionType = ::PortableDeviceMultiCollection<ALPAKA_ACCELERATOR_NAMESPACE::Device, T0, Args...>;
  };
#endif

}  // namespace traits

#endif  // DataFormats_Portable_interface_alpaka_PortableCollection_h
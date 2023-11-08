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

#endif  // DataFormats_Portable_interface_alpaka_PortableCollection_h

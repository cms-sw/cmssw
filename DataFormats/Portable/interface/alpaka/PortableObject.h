#ifndef DataFormats_Portable_interface_alpaka_PortableObject_h
#define DataFormats_Portable_interface_alpaka_PortableObject_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// This header is not used by PortableObject, but is included here to automatically
// provide its content to users of ALPAKA_ACCELERATOR_NAMESPACE::PortableObject.
#include "HeterogeneousCore/AlpakaInterface/interface/AssertDeviceMatchesHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // generic struct-based product in the device (that may be host) memory
  template <typename T>
  using PortableObject = ::PortableObject<T, Device>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_Portable_interface_alpaka_PortableObject_h

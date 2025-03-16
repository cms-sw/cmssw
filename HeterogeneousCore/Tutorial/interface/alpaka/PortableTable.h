#ifndef HeterogeneousCore_Tutorial_interface_alpaka_PortableTable_h
#define HeterogeneousCore_Tutorial_interface_alpaka_PortableTable_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/Tutorial/interface/PortableTable.h"

// This header is not used by PortableTable, but is included here to automatically
// provide its content to users of ALPAKA_ACCELERATOR_NAMESPACE::tutrial::PortableTable.
#include "HeterogeneousCore/AlpakaInterface/interface/AssertDeviceMatchesHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial {

  // generic struct-based product in the device (that may be host) memory
  using PortableTable = ::tutorial::PortableTable<Device>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial

#endif  // HeterogeneousCore_Tutorial_interface_alpaka_PortableTable_h

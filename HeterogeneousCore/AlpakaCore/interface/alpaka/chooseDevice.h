#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_chooseDevice_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_chooseDevice_h

#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::detail {
  Device const& chooseDevice(edm::StreamID id);
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::detail

#endif  // HeterogeneousCore_AlpakaCore_interface_chooseDevice_h

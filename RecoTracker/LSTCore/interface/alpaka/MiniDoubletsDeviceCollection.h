#ifndef RecoTracker_LSTCore_interface_alpaka_MiniDoubletsDeviceCollection_h
#define RecoTracker_LSTCore_interface_alpaka_MiniDoubletsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using MiniDoubletsDeviceCollection = PortableCollection<MiniDoubletsSoABlocks>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif

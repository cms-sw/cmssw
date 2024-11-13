#ifndef RecoTracker_LSTCore_interface_alpaka_ObjectRangesDeviceCollection_h
#define RecoTracker_LSTCore_interface_alpaka_ObjectRangesDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using ObjectRangesDeviceCollection = PortableCollection<ObjectRangesSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif

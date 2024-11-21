#ifndef RecoTracker_LSTCore_interface_MiniDoubletsHostCollection_h
#define RecoTracker_LSTCore_interface_MiniDoubletsHostCollection_h

#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using MiniDoubletsHostCollection = PortableHostMultiCollection<MiniDoubletsSoA, MiniDoubletsOccupancySoA>;
}  // namespace lst
#endif

#ifndef RecoTracker_LSTCore_interface_QuintupletsHostCollection_h
#define RecoTracker_LSTCore_interface_QuintupletsHostCollection_h

#include "RecoTracker/LSTCore/interface/QuintupletsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using QuintupletsHostCollection = PortableHostMultiCollection<QuintupletsSoA, QuintupletsOccupancySoA>;
}  // namespace lst
#endif

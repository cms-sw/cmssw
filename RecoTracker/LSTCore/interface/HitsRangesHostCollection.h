#ifndef RecoTracker_LSTCore_interface_HitsRangesHostCollection_h
#define RecoTracker_LSTCore_interface_HitsRangesHostCollection_h

#include "RecoTracker/LSTCore/interface/HitsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using HitsRangesHostCollection = PortableHostCollection<HitsRangesSoA>;
}  // namespace lst
#endif

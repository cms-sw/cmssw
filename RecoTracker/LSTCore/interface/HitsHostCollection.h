#ifndef RecoTracker_LSTCore_interface_HitsHostCollection_h
#define RecoTracker_LSTCore_interface_HitsHostCollection_h

#include "RecoTracker/LSTCore/interface/HitsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using HitsHostCollection = PortableHostMultiCollection<HitsSoA, PixelHitsSoA>;
}  // namespace lst
#endif

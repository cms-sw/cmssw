#ifndef RecoTracker_LSTCore_interface_PixelQuintupletsHostCollection_h
#define RecoTracker_LSTCore_interface_PixelQuintupletsHostCollection_h

#include "RecoTracker/LSTCore/interface/PixelQuintupletsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using PixelQuintupletsHostCollection = PortableHostCollection<PixelQuintupletsSoA>;
}  // namespace lst
#endif

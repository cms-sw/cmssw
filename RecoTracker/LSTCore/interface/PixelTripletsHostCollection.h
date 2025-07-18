#ifndef RecoTracker_LSTCore_interface_PixelTripletsHostCollection_h
#define RecoTracker_LSTCore_interface_PixelTripletsHostCollection_h

#include "RecoTracker/LSTCore/interface/PixelTripletsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using PixelTripletsHostCollection = PortableHostCollection<PixelTripletsSoA>;
}  // namespace lst
#endif

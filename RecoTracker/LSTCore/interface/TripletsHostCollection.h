#ifndef RecoTracker_LSTCore_interface_TripletsHostCollection_h
#define RecoTracker_LSTCore_interface_TripletsHostCollection_h

#include "RecoTracker/LSTCore/interface/TripletsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using TripletsHostCollection = PortableHostMultiCollection<TripletsSoA, TripletsOccupancySoA>;
}  // namespace lst
#endif

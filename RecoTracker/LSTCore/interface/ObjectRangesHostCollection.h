#ifndef RecoTracker_LSTCore_interface_ObjectRangesHostCollection_h
#define RecoTracker_LSTCore_interface_ObjectRangesHostCollection_h

#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using ObjectRangesHostCollection = PortableHostCollection<ObjectRangesSoA>;
}  // namespace lst
#endif

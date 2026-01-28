#ifndef RecoTracker_LSTCore_interface_QuadrupletsHostCollection_h
#define RecoTracker_LSTCore_interface_QuadrupletsHostCollection_h

#include "RecoTracker/LSTCore/interface/QuadrupletsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using QuadrupletsHostCollection = PortableHostCollection<QuadrupletsSoABlocks>;
}  // namespace lst
#endif

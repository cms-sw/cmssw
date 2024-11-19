#ifndef RecoTracker_LSTCore_interface_EndcapGeometryDevHostCollection_h
#define RecoTracker_LSTCore_interface_EndcapGeometryDevHostCollection_h

#include "RecoTracker/LSTCore/interface/EndcapGeometryDevSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using EndcapGeometryDevHostCollection = PortableHostCollection<EndcapGeometryDevSoA>;
}  // namespace lst
#endif

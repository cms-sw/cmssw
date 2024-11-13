#ifndef RecoTracker_LSTCore_interface_alpaka_EndcapGeometryDevDeviceCollection_h
#define RecoTracker_LSTCore_interface_alpaka_EndcapGeometryDevDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometryDevSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using EndcapGeometryDevDeviceCollection = PortableCollection<EndcapGeometryDevSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif

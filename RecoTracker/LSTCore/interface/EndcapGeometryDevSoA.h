#ifndef RecoTracker_LSTCore_interface_EndcapGeometryDevSoA_h
#define RecoTracker_LSTCore_interface_EndcapGeometryDevSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

namespace lst {

  GENERATE_SOA_LAYOUT(EndcapGeometryDevSoALayout, SOA_COLUMN(unsigned int, geoMapDetId), SOA_COLUMN(float, geoMapPhi))

  using EndcapGeometryDevSoA = EndcapGeometryDevSoALayout<>;

  using EndcapGeometryDev = EndcapGeometryDevSoA::View;
  using EndcapGeometryDevConst = EndcapGeometryDevSoA::ConstView;

}  // namespace lst

#endif

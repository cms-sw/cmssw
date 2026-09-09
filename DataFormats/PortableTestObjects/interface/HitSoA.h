#ifndef DataFormats_PortableTestObjects_interface_HitSoA_h
#define DataFormats_PortableTestObjects_interface_HitSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace portabletest {

  GENERATE_SOA_LAYOUT(HitLayout, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z))
  GENERATE_SOA_LAYOUT(HitToTrackLayout, SOA_COLUMN(int, trackIndex))
  GENERATE_SOA_LAYOUT(TrackBeginLayout, SOA_COLUMN(int, trackBegin))

  using HitSoA = HitLayout<>;
  using HitToTrackSoA = HitToTrackLayout<>;
  using TrackBeginSoA = TrackBeginLayout<>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_HitSoA_h

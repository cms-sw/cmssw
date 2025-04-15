#ifndef RecoTracker_LSTCore_interface_PixelSegmentsSoA_h
#define RecoTracker_LSTCore_interface_PixelSegmentsSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {

  GENERATE_SOA_LAYOUT(PixelSegmentsSoALayout,
                      SOA_COLUMN(uint4, pLSHitsIdxs),
                      SOA_COLUMN(char, isDup),
                      SOA_COLUMN(bool, partOfPT5),
                      SOA_COLUMN(float, score),
                      SOA_COLUMN(float, circleCenterX),
                      SOA_COLUMN(float, circleCenterY),
                      SOA_COLUMN(float, circleRadius))

  using PixelSegmentsSoA = PixelSegmentsSoALayout<>;

  using PixelSegments = PixelSegmentsSoA::View;
  using PixelSegmentsConst = PixelSegmentsSoA::ConstView;

}  // namespace lst

#endif

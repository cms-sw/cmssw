#ifndef RecoTracker_LSTCore_interface_HitsSoA_h
#define RecoTracker_LSTCore_interface_HitsSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {

  GENERATE_SOA_LAYOUT(HitsExtendedSoALayout,
                      SOA_COLUMN(uint16_t, moduleIndices),
                      SOA_COLUMN(float, rts),
                      SOA_COLUMN(float, phis),
                      SOA_COLUMN(float, etas),
                      SOA_COLUMN(float, highEdgeXs),
                      SOA_COLUMN(float, highEdgeYs),
                      SOA_COLUMN(float, lowEdgeXs),
                      SOA_COLUMN(float, lowEdgeYs))

  GENERATE_SOA_LAYOUT(HitsRangesSoALayout,
                      SOA_COLUMN(ArrayIx2, hitRanges),
                      SOA_COLUMN(int, hitRangesLower),
                      SOA_COLUMN(int, hitRangesUpper),
                      SOA_COLUMN(int16_t, hitRangesnLower),
                      SOA_COLUMN(int16_t, hitRangesnUpper))

  using HitsExtendedSoA = HitsExtendedSoALayout<>;
  using HitsRangesSoA = HitsRangesSoALayout<>;

  using HitsExtended = HitsExtendedSoA::View;
  using HitsExtendedConst = HitsExtendedSoA::ConstView;
  using HitsRanges = HitsRangesSoA::View;
  using HitsRangesConst = HitsRangesSoA::ConstView;

}  // namespace lst

#endif

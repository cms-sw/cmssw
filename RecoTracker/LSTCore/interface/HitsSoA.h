#ifndef RecoTracker_LSTCore_interface_HitsSoA_h
#define RecoTracker_LSTCore_interface_HitsSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {

  GENERATE_SOA_LAYOUT(HitsSoALayout,
                      SOA_COLUMN(float, xs),
                      SOA_COLUMN(float, ys),
                      SOA_COLUMN(float, zs),
                      SOA_COLUMN(uint16_t, moduleIndices),
                      SOA_COLUMN(unsigned int, idxs),
                      SOA_COLUMN(unsigned int, detid),
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
                      SOA_COLUMN(int8_t, hitRangesnLower),
                      SOA_COLUMN(int8_t, hitRangesnUpper))

  using HitsSoA = HitsSoALayout<>;
  using HitsRangesSoA = HitsRangesSoALayout<>;

  using Hits = HitsSoA::View;
  using HitsConst = HitsSoA::ConstView;
  using HitsRanges = HitsRangesSoA::View;
  using HitsRangesConst = HitsRangesSoA::ConstView;

}  // namespace lst

#endif

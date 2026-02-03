#ifndef RecoTracker_LSTCore_interface_HitsSoA_h
#define RecoTracker_LSTCore_interface_HitsSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
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

  GENERATE_SOA_BLOCKS(HitsSoALayout, SOA_BLOCK(extended, HitsExtendedSoALayout), SOA_BLOCK(ranges, HitsRangesSoALayout))

  using HitsExtendedSoA = HitsExtendedSoALayout<>;
  using HitsRangesSoA = HitsRangesSoALayout<>;
  using HitsSoA = HitsSoALayout<>;

  using HitsExtended = HitsExtendedSoA::View;
  using HitsExtendedConst = HitsExtendedSoA::ConstView;
  using HitsRanges = HitsRangesSoA::View;
  using HitsRangesConst = HitsRangesSoA::ConstView;
  using HitsView = HitsSoA::View;
  using HitsConstView = HitsSoA::ConstView;

  // Template based accessor for getting specific SoA views. Needed in LSTEvent.dev.cc
  template <typename TSoA>
  struct HitsViewAccessor;

  template <>
  struct HitsViewAccessor<HitsExtendedSoA> {
    static constexpr auto get(auto const& v) { return v.extended(); }
  };

  template <>
  struct HitsViewAccessor<HitsRangesSoA> {
    static constexpr auto get(auto const& v) { return v.ranges(); }
  };

}  // namespace lst

#endif

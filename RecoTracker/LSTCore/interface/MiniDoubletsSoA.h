#ifndef RecoTracker_LSTCore_interface_MiniDoubletsSoA_h
#define RecoTracker_LSTCore_interface_MiniDoubletsSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

namespace lst {

  GENERATE_SOA_LAYOUT(MiniDoubletsSoALayout,
                      SOA_COLUMN(unsigned int, anchorHitIndices),
                      SOA_COLUMN(unsigned int, outerHitIndices),
                      SOA_COLUMN(float, anchorX),
                      SOA_COLUMN(float, anchorY),
                      SOA_COLUMN(float, anchorZ),
                      SOA_COLUMN(float, anchorRt),
                      SOA_COLUMN(float, anchorPhi),
                      SOA_COLUMN(float, anchorEta),
                      SOA_COLUMN(float, outerX),
                      SOA_COLUMN(float, outerY),
#ifdef CUT_VALUE_DEBUG
                      SOA_COLUMN(float, outerRt),
                      SOA_COLUMN(float, outerPhi),
                      SOA_COLUMN(float, outerEta),
                      SOA_COLUMN(float, shiftedXs),
                      SOA_COLUMN(float, shiftedYs),
                      SOA_COLUMN(float, shiftedZs),
                      SOA_COLUMN(float, noShiftedDphis),
                      SOA_COLUMN(float, noShiftedDphiChanges),
#endif
                      SOA_COLUMN(float, outerZ))

  // Build-only MD columns (MD creation -> segment creation), in their own collection released after the LS stage
  GENERATE_SOA_LAYOUT(MiniDoubletsBuildSoALayout,
                      SOA_COLUMN(float, dphichanges),
                      SOA_COLUMN(float, dzs),
                      SOA_COLUMN(float, dphis),
                      SOA_COLUMN(unsigned int, connectedMax))

  using MiniDoubletsBuildSoA = MiniDoubletsBuildSoALayout<>;
  using MiniDoubletsBuild = MiniDoubletsBuildSoA::View;
  using MiniDoubletsBuildConst = MiniDoubletsBuildSoA::ConstView;

  GENERATE_SOA_LAYOUT(MiniDoubletsOccupancySoALayout,
                      SOA_COLUMN(unsigned int, nMDs),
                      SOA_COLUMN(unsigned int, totOccupancyMDs))

  GENERATE_SOA_LAYOUT(QuintupletsRangesSoALayout, SOA_COLUMN(uint32_t, offset), SOA_COLUMN(uint32_t, n));

  using QuintupletsRangesSoA = QuintupletsRangesSoALayout<>;
  using QuintupletsRanges = QuintupletsRangesSoA::View;
  using QuintupletsRangesConst = QuintupletsRangesSoA::ConstView;

  // Per-MD T5 counters and T5-by-MD ranges, in a collection that lives only in the T5 stage
  GENERATE_SOA_LAYOUT(MiniDoubletsT5CountsSoALayout,
                      SOA_COLUMN(unsigned int, connectedT5s0Max),
                      SOA_COLUMN(unsigned int, connectedT5s1Max))

  GENERATE_SOA_BLOCKS(MiniDoubletsT5BuildSoABlocksLayout,
                      SOA_BLOCK(t5Counts, MiniDoubletsT5CountsSoALayout),
                      SOA_BLOCK(quintupletsRangesByMD0, QuintupletsRangesSoALayout),
                      SOA_BLOCK(quintupletsRangesByMD1, QuintupletsRangesSoALayout))

  using MiniDoubletsT5CountsSoA = MiniDoubletsT5CountsSoALayout<>;
  using MiniDoubletsT5Counts = MiniDoubletsT5CountsSoA::View;
  using MiniDoubletsT5CountsConst = MiniDoubletsT5CountsSoA::ConstView;
  using MiniDoubletsT5BuildSoABlocks = MiniDoubletsT5BuildSoABlocksLayout<>;

  GENERATE_SOA_BLOCKS(MiniDoubletsSoABlocksLayout,
                      SOA_BLOCK(miniDoublets, MiniDoubletsSoALayout),
                      SOA_BLOCK(miniDoubletsOccupancy, MiniDoubletsOccupancySoALayout))

  using MiniDoubletsSoA = MiniDoubletsSoALayout<>;
  using MiniDoubletsOccupancySoA = MiniDoubletsOccupancySoALayout<>;
  using MiniDoubletsSoABlocks = MiniDoubletsSoABlocksLayout<>;

  using MiniDoublets = MiniDoubletsSoA::View;
  using MiniDoubletsConst = MiniDoubletsSoA::ConstView;
  using MiniDoubletsOccupancy = MiniDoubletsOccupancySoA::View;
  using MiniDoubletsOccupancyConst = MiniDoubletsOccupancySoA::ConstView;
  using MiniDoubletsSoABlocksView = MiniDoubletsSoABlocks::View;
  using MiniDoubletsSoABlocksConstView = MiniDoubletsSoABlocks::ConstView;

  // Template based accessor for getting specific SoA views. Needed in LSTEvent.dev.cc
  template <typename TSoA>
  struct MiniDoubletsViewAccessor;

  template <>
  struct MiniDoubletsViewAccessor<MiniDoubletsSoA> {
    static constexpr auto get(auto const& v) { return v.miniDoublets(); }
  };

  template <>
  struct MiniDoubletsViewAccessor<MiniDoubletsOccupancySoA> {
    static constexpr auto get(auto const& v) { return v.miniDoubletsOccupancy(); }
  };

}  // namespace lst

#endif

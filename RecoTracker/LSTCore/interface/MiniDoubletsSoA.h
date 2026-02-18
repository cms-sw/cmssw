#ifndef RecoTracker_LSTCore_interface_MiniDoubletsSoA_h
#define RecoTracker_LSTCore_interface_MiniDoubletsSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

namespace lst {

  GENERATE_SOA_LAYOUT(MiniDoubletsSoALayout,
                      SOA_COLUMN(unsigned int, anchorHitIndices),
                      SOA_COLUMN(unsigned int, outerHitIndices),
                      SOA_COLUMN(uint16_t, moduleIndices),
                      SOA_COLUMN(float, dphichanges),
                      SOA_COLUMN(float, dzs),
                      SOA_COLUMN(float, dphis),
                      SOA_COLUMN(float, anchorX),
                      SOA_COLUMN(float, anchorY),
                      SOA_COLUMN(float, anchorZ),
                      SOA_COLUMN(float, anchorRt),
                      SOA_COLUMN(float, anchorPhi),
                      SOA_COLUMN(float, anchorEta),
                      SOA_COLUMN(float, anchorHighEdgeX),
                      SOA_COLUMN(float, anchorHighEdgeY),
                      SOA_COLUMN(float, anchorLowEdgeX),
                      SOA_COLUMN(float, anchorLowEdgeY),
                      SOA_COLUMN(float, anchorLowEdgePhi),
                      SOA_COLUMN(float, anchorHighEdgePhi),
                      SOA_COLUMN(float, outerX),
                      SOA_COLUMN(float, outerY),
                      SOA_COLUMN(float, outerZ),
#ifdef CUT_VALUE_DEBUG
                      SOA_COLUMN(float, outerRt),
                      SOA_COLUMN(float, outerPhi),
                      SOA_COLUMN(float, outerEta),
                      SOA_COLUMN(float, outerHighEdgeX),
                      SOA_COLUMN(float, outerHighEdgeY),
                      SOA_COLUMN(float, outerLowEdgeX),
                      SOA_COLUMN(float, outerLowEdgeY),
                      SOA_COLUMN(float, shiftedXs),
                      SOA_COLUMN(float, shiftedYs),
                      SOA_COLUMN(float, shiftedZs),
                      SOA_COLUMN(float, noShiftedDphis),
                      SOA_COLUMN(float, noShiftedDphiChanges),
#endif
                      SOA_COLUMN(unsigned int, connectedMax))

  GENERATE_SOA_LAYOUT(MiniDoubletsOccupancySoALayout,
                      SOA_COLUMN(unsigned int, nMDs),
                      SOA_COLUMN(unsigned int, totOccupancyMDs))

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

#ifndef RecoTracker_LSTCore_interface_SegmentsSoA_h
#define RecoTracker_LSTCore_interface_SegmentsSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {

  GENERATE_SOA_LAYOUT(SegmentsSoALayout,
                      SOA_COLUMN(FPX, dPhis),
                      SOA_COLUMN(FPX, dPhiMins),
                      SOA_COLUMN(FPX, dPhiMaxs),
                      SOA_COLUMN(FPX, dPhiChanges),
                      SOA_COLUMN(FPX, dPhiChangeMins),
                      SOA_COLUMN(FPX, dPhiChangeMaxs),
#ifdef CUT_VALUE_DEBUG
                      SOA_COLUMN(FPX, zHis),
                      SOA_COLUMN(FPX, zLos),
                      SOA_COLUMN(FPX, rtHis),
                      SOA_COLUMN(FPX, rtLos),
                      SOA_COLUMN(FPX, dAlphaInners),
                      SOA_COLUMN(FPX, dAlphaOuters),
                      SOA_COLUMN(FPX, dAlphaInnerOuters),
#endif
                      SOA_COLUMN(uint16_t, innerLowerModuleIndices),
                      SOA_COLUMN(uint16_t, outerLowerModuleIndices),
                      SOA_COLUMN(Params_LS::ArrayUxLayers, mdIndices),
                      SOA_COLUMN(unsigned int, innerMiniDoubletAnchorHitIndices),
                      SOA_COLUMN(unsigned int, outerMiniDoubletAnchorHitIndices),
                      SOA_COLUMN(unsigned int, connectedMax))

  GENERATE_SOA_LAYOUT(SegmentsOccupancySoALayout,
                      SOA_COLUMN(unsigned int, nSegments),  //number of segments per inner lower module
                      SOA_COLUMN(unsigned int, totOccupancySegments))

  GENERATE_SOA_BLOCKS(SegmentsSoABlocksLayout,
                      SOA_BLOCK(segments, SegmentsSoALayout),
                      SOA_BLOCK(segmentsOccupancy, SegmentsOccupancySoALayout))

  using SegmentsSoA = SegmentsSoALayout<>;
  using SegmentsOccupancySoA = SegmentsOccupancySoALayout<>;

  using Segments = SegmentsSoA::View;
  using SegmentsConst = SegmentsSoA::ConstView;
  using SegmentsOccupancy = SegmentsOccupancySoA::View;
  using SegmentsOccupancyConst = SegmentsOccupancySoA::ConstView;

  using SegmentsSoABlocks = SegmentsSoABlocksLayout<>;
  using SegmentsSoABlocksView = SegmentsSoABlocks::View;
  using SegmentsSoABlocksConstView = SegmentsSoABlocks::ConstView;

  // Template based accessor for getting specific SoA views. Needed in LSTEvent.dev.cc
  template <typename TSoA>
  struct SegmentsViewAccessor;

  template <>
  struct SegmentsViewAccessor<SegmentsSoA> {
    static constexpr auto get(auto const& v) { return v.segments(); }
  };

  template <>
  struct SegmentsViewAccessor<SegmentsOccupancySoA> {
    static constexpr auto get(auto const& v) { return v.segmentsOccupancy(); }
  };

}  // namespace lst

#endif

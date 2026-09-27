#ifndef RecoTracker_LSTCore_interface_SegmentsSoA_h
#define RecoTracker_LSTCore_interface_SegmentsSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {

  GENERATE_SOA_LAYOUT(SegmentsSoALayout,
                      SOA_COLUMN(FPX, dPhiChanges),
                      SOA_COLUMN(FPX, dPhiChangeMins),
                      SOA_COLUMN(FPX, dPhiChangeMaxs),
#ifdef CUT_VALUE_DEBUG
                      SOA_COLUMN(FPX, dPhis),
                      SOA_COLUMN(FPX, dPhiMins),
                      SOA_COLUMN(FPX, dPhiMaxs),
                      SOA_COLUMN(FPX, zHis),
                      SOA_COLUMN(FPX, zLos),
                      SOA_COLUMN(FPX, rtHis),
                      SOA_COLUMN(FPX, rtLos),
                      SOA_COLUMN(FPX, dAlphaInners),
                      SOA_COLUMN(FPX, dAlphaOuters),
                      SOA_COLUMN(FPX, dAlphaInnerOuters),
#endif
                      SOA_COLUMN(uint16_t, outerLowerModuleIndices),
                      SOA_COLUMN(Params_LS::ArrayUxLayers, mdIndices))

  // Per-segment T3 counter (T3 count -> T3 create), in a collection that lives only in the T3 stage
  GENERATE_SOA_LAYOUT(SegmentsT3CountsSoALayout, SOA_COLUMN(unsigned int, connectedMax))

  using SegmentsT3CountsSoA = SegmentsT3CountsSoALayout<>;
  using SegmentsT3Counts = SegmentsT3CountsSoA::View;
  using SegmentsT3CountsConst = SegmentsT3CountsSoA::ConstView;

  GENERATE_SOA_LAYOUT(SegmentsOccupancySoALayout,
                      SOA_COLUMN(unsigned int, nSegments),  //number of segments per inner lower module
                      SOA_COLUMN(unsigned int, totOccupancySegments))

  GENERATE_SOA_BLOCKS(SegmentsSoABlocksLayout,
                      SOA_BLOCK(segments, SegmentsSoALayout),
                      SOA_BLOCK(segmentsOccupancy, SegmentsOccupancySoALayout))

  // Loose-sized scratch filled by CreateSegments, 5 B per slot: the MD pair of each produced segment as its index
  // in the inner x outer MD product of the module pair, and the outer module as its slot in the inner module's map.
  GENERATE_SOA_LAYOUT(SegmentCandidatesSoALayout,
                      SOA_COLUMN(uint32_t, mdPairIndices),
                      SOA_COLUMN(uint8_t, connectedModuleSlots))
  static_assert(max_connected_modules <= 256, "connectedModuleSlots is a uint8_t");

  GENERATE_SOA_BLOCKS(SegmentCandidatesSoABlocksLayout,
                      SOA_BLOCK(candidates, SegmentCandidatesSoALayout),
                      SOA_BLOCK(segmentsOccupancy, SegmentsOccupancySoALayout))

  using SegmentsSoA = SegmentsSoALayout<>;
  using SegmentsOccupancySoA = SegmentsOccupancySoALayout<>;

  using Segments = SegmentsSoA::View;
  using SegmentsConst = SegmentsSoA::ConstView;
  using SegmentsOccupancy = SegmentsOccupancySoA::View;
  using SegmentsOccupancyConst = SegmentsOccupancySoA::ConstView;

  using SegmentCandidatesSoA = SegmentCandidatesSoALayout<>;
  using SegmentCandidates = SegmentCandidatesSoA::View;
  using SegmentCandidatesConst = SegmentCandidatesSoA::ConstView;
  using SegmentCandidatesSoABlocks = SegmentCandidatesSoABlocksLayout<>;

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

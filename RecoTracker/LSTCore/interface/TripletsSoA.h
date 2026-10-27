#ifndef RecoTracker_LSTCore_interface_TripletsSoA_h
#define RecoTracker_LSTCore_interface_TripletsSoA_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {
  GENERATE_SOA_LAYOUT(TripletsSoALayout,
                      SOA_COLUMN(ArrayUx2, segmentIndices),                        // inner and outer segment indices
                      SOA_COLUMN(Params_T3::ArrayU16xLayers, lowerModuleIndices),  // lower module index in each layer
                      SOA_COLUMN(float, centerX),         // lower/anchor-hit based circle center x
                      SOA_COLUMN(float, centerY),         // lower/anchor-hit based circle center y
                      SOA_COLUMN(float, radius),          // lower/anchor-hit based circle radius
                      SOA_COLUMN(float, fakeScore),       // DNN confidence score for fake t3
                      SOA_COLUMN(float, promptScore),     // DNN confidence score for real (prompt) t3
                      SOA_COLUMN(float, displacedScore),  // DNN confidence score for real (displaced) t3
                      SOA_COLUMN(int8_t, charge),         // +-1
#ifdef CUT_VALUE_DEBUG
                      SOA_COLUMN(FPX, betaIn),  // beta/chord angle of the inner segment
                      SOA_COLUMN(float, betaInCut),
#endif
                      SOA_COLUMN(bool, partOfPT5),  // is it used in a pT5
                      SOA_COLUMN(bool, partOfT5),   // is it used in a T5
                      SOA_COLUMN(bool, partOfPT3),  // is it used in a pT3
                      SOA_COLUMN(uint8_t, flags));  // T3Flag bits

  using TripletsSoA = TripletsSoALayout<>;
  using Triplets = TripletsSoA::View;
  using TripletsConst = TripletsSoA::ConstView;

  GENERATE_SOA_LAYOUT(TripletsOccupancySoALayout,
                      SOA_COLUMN(unsigned int, nTriplets),
                      SOA_COLUMN(unsigned int, totOccupancyTriplets));

  using TripletsOccupancySoA = TripletsOccupancySoALayout<>;
  using TripletsOccupancy = TripletsOccupancySoA::View;
  using TripletsOccupancyConst = TripletsOccupancySoA::ConstView;

  GENERATE_SOA_LAYOUT(TripletsRangesSoALayout, SOA_COLUMN(int, offset), SOA_COLUMN(uint32_t, n));

  using TripletsRangesSoA = TripletsRangesSoALayout<>;
  using TripletsRanges = TripletsRangesSoA::View;
  using TripletsRangesConst = TripletsRangesSoA::ConstView;

  // index and fast cached data if any
  GENERATE_SOA_LAYOUT(TripletsBySegmentSoALayout, SOA_COLUMN(unsigned int, tripletIndex));

  using TripletsBySegmentSoA = TripletsBySegmentSoALayout<>;
  using TripletsBySegment = TripletsBySegmentSoA::View;
  using TripletsBySegmentConst = TripletsBySegmentSoA::ConstView;

  // index and fast cached data if any
  GENERATE_SOA_LAYOUT(TripletsByMDSoALayout, SOA_COLUMN(unsigned int, tripletIndex));

  using TripletsByMDSoA = TripletsByMDSoALayout<>;
  using TripletsByMD = TripletsByMDSoA::View;
  using TripletsByMDConst = TripletsByMDSoA::ConstView;

  GENERATE_SOA_BLOCKS(TripletsSoABlocksLayout,
                      SOA_BLOCK(triplets, TripletsSoALayout),
                      SOA_BLOCK(tripletsOccupancy, TripletsOccupancySoALayout),
                      SOA_BLOCK(tripletsBySegment, TripletsBySegmentSoALayout),
                      SOA_BLOCK(tripletsByMD, TripletsByMDSoALayout))

  using TripletsSoABlocks = TripletsSoABlocksLayout<>;
  using TripletsSoABlocksView = TripletsSoABlocks::View;
  using TripletsSoABlocksConstView = TripletsSoABlocks::ConstView;

  // Per-segment and per-MD ranges of the by-segment/by-MD triplet lists, sized by segments and MDs,
  // kept apart from the triplet-sized blocks so that the triplets can be compacted after creation.
  GENERATE_SOA_BLOCKS(TripletsListRangesSoABlocksLayout,
                      SOA_BLOCK(tripletsRangesBySegment, TripletsRangesSoALayout),
                      SOA_BLOCK(tripletsRangesByMD, TripletsRangesSoALayout))

  using TripletsListRangesSoABlocks = TripletsListRangesSoABlocksLayout<>;

  // Creation-time buffers, sized by the loose count and freed after CompactTriplets: the builder's step-1
  // candidate segment pairs with their flags, and the columns the builder computes.
  GENERATE_SOA_LAYOUT(TripletsScratchSoALayout, SOA_COLUMN(ArrayUx2, segmentIndices), SOA_COLUMN(uint8_t, flags));

  using TripletsScratch = TripletsScratchSoALayout<>::View;

  GENERATE_SOA_LAYOUT(TripletsBuildSoALayout,
                      SOA_COLUMN(ArrayUx2, segmentIndices),
                      SOA_COLUMN(float, centerX),
                      SOA_COLUMN(float, centerY),
                      SOA_COLUMN(float, radius),
                      SOA_COLUMN(float, fakeScore),
                      SOA_COLUMN(float, promptScore),
                      SOA_COLUMN(float, displacedScore),
                      SOA_COLUMN(int8_t, charge),
#ifdef CUT_VALUE_DEBUG
                      SOA_COLUMN(FPX, betaIn),
                      SOA_COLUMN(float, betaInCut),
#endif
                      SOA_COLUMN(uint8_t, flags));

  using TripletsBuildSoA = TripletsBuildSoALayout<>;
  using TripletsBuild = TripletsBuildSoA::View;
  using TripletsBuildConst = TripletsBuildSoA::ConstView;

  GENERATE_SOA_BLOCKS(TripletsBuildSoABlocksLayout,
                      SOA_BLOCK(triplets, TripletsBuildSoALayout),
                      SOA_BLOCK(tripletsOccupancy, TripletsOccupancySoALayout),
                      SOA_BLOCK(scratch, TripletsScratchSoALayout))

  using TripletsBuildSoABlocks = TripletsBuildSoABlocksLayout<>;

  // Template based accessor for getting specific SoA views. Needed in LSTEvent.dev.cc
  template <typename TSoA>
  struct TripletsViewAccessor;

  template <>
  struct TripletsViewAccessor<TripletsSoA> {
    static constexpr auto get(auto const& v) { return v.triplets(); }
  };

  template <>
  struct TripletsViewAccessor<TripletsOccupancySoA> {
    static constexpr auto get(auto const& v) { return v.tripletsOccupancy(); }
  };

}  // namespace lst
#endif

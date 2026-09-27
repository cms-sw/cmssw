#ifndef RecoTracker_LSTCore_interface_QuintupletsSoA_h
#define RecoTracker_LSTCore_interface_QuintupletsSoA_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {
  GENERATE_SOA_LAYOUT(QuintupletsSoALayout,
                      SOA_COLUMN(ArrayUx2, tripletIndices),                        // inner and outer triplet indices
                      SOA_COLUMN(Params_T5::ArrayU16xLayers, lowerModuleIndices),  // lower module index in each layer
                      SOA_COLUMN(Params_T5::ArrayU8xLayers, logicalLayers),        // layer ID
                      SOA_COLUMN(Params_T5::ArrayUxHits, hitIndices),              // hit indices
                      SOA_COLUMN(Params_T5::ArrayFxEmbed, t5Embed),                // t5 embedding vector
                      SOA_COLUMN(FPX, eta),
                      SOA_COLUMN(FPX, phi),
                      SOA_COLUMN(char, isDup),            // duplicate flag
                      SOA_COLUMN(unsigned int, nLayers),  // number of active layers (5 base)
                      SOA_COLUMN(bool, partOfPT5),
                      SOA_COLUMN(float, regressionRadius),
                      SOA_COLUMN(float, regressionCenterX),
                      SOA_COLUMN(float, regressionCenterY),
                      SOA_COLUMN(float, dnnScore),
#ifdef CUT_VALUE_DEBUG
                      SOA_COLUMN(FPX, bridgeRadius),    // "middle"/bridge triplet radius
                      SOA_COLUMN(float, rzChiSquared),  // r-z only chi2
                      SOA_COLUMN(float, chiSquared),
                      SOA_COLUMN(float, dBeta1),
                      SOA_COLUMN(float, dBeta2),
                      SOA_COLUMN(float, mdDirMeanW),  // T5 DNN extra inputs: MD direction, mean and largest
                      SOA_COLUMN(float, mdDirMaxW),
                      SOA_COLUMN(float, nT3OutMid),    // T3s leaving the middle MD
                      SOA_COLUMN(float, nT3OutFirst),  // T3s leaving the first MD
                      SOA_COLUMN(float, nMDFirstMod),  // MDs in the first module
                      SOA_COLUMN(float, dcaXY),
#endif
                      SOA_COLUMN(FPX, innerRadius));  // inner triplet circle radius (outer: triplets.radius of T3 1)

  using QuintupletsSoA = QuintupletsSoALayout<>;
  using Quintuplets = QuintupletsSoA::View;
  using QuintupletsConst = QuintupletsSoA::ConstView;

  GENERATE_SOA_LAYOUT(QuintupletsOccupancySoALayout,
                      SOA_COLUMN(unsigned int, nQuintuplets),
                      SOA_COLUMN(unsigned int, totOccupancyQuintuplets));

  using QuintupletsOccupancySoA = QuintupletsOccupancySoALayout<>;
  using QuintupletsOccupancy = QuintupletsOccupancySoA::View;
  using QuintupletsOccupancyConst = QuintupletsOccupancySoA::ConstView;

  // index and fast cached data if any
  GENERATE_SOA_LAYOUT(QuintupletsBySegmentSoALayout, SOA_COLUMN(unsigned int, quintupletIndex));

  using QuintupletsBySegmentSoA = QuintupletsBySegmentSoALayout<>;
  using QuintupletsBySegment = QuintupletsBySegmentSoA::View;
  using QuintupletsBySegmentConst = QuintupletsBySegmentSoA::ConstView;

  // index and fast cached data if any
  GENERATE_SOA_LAYOUT(QuintupletsByMDSoALayout,
                      SOA_COLUMN(unsigned int, quintupletIndex),
                      SOA_COLUMN(uint32_t, mdBarCode));
  constexpr uint32_t kT5ByMDBarCodeMask = 0xFF;
  constexpr short kT5ByMDBarOffset = 8;

  using QuintupletsByMDSoA = QuintupletsByMDSoALayout<>;
  using QuintupletsByMD = QuintupletsByMDSoA::View;
  using QuintupletsByMDConst = QuintupletsByMDSoA::ConstView;

  // Build-time record of a selected quintuplet, one per counting-kernel slot; the full quintuplet is written
  // afterwards to an exactly sized QuintupletsSoA.
  GENERATE_SOA_LAYOUT(QuintupletsLooseSoALayout,
                      SOA_COLUMN(ArrayUx2, preAllocatedTripletIndices),  // candidate pairs (step 1)
                      SOA_COLUMN(ArrayUx2, tripletIndices),              // selected pairs (step 2)
                      SOA_COLUMN(ArrayUx2, byMDIndices),                 // slots in quintupletsByMD0/1
                      SOA_COLUMN(float, bridgeRadius),
#ifdef CUT_VALUE_DEBUG
                      SOA_COLUMN(float, rzChiSquared),
                      SOA_COLUMN(float, dBeta1),
                      SOA_COLUMN(float, dBeta2),
#endif
                      SOA_COLUMN(float, dnnScore));

  using QuintupletsLooseSoA = QuintupletsLooseSoALayout<>;
  using QuintupletsLoose = QuintupletsLooseSoA::View;
  using QuintupletsLooseConst = QuintupletsLooseSoA::ConstView;

  GENERATE_SOA_BLOCKS(QuintupletsSoABlocksLayout,
                      SOA_BLOCK(quintuplets, QuintupletsSoALayout),
                      SOA_BLOCK(quintupletsOccupancy, QuintupletsOccupancySoALayout),
                      SOA_BLOCK(quintupletsByMD0, QuintupletsByMDSoALayout),
                      SOA_BLOCK(quintupletsByMD1, QuintupletsByMDSoALayout))

  using QuintupletsSoABlocks = QuintupletsSoABlocksLayout<>;
  using QuintupletsSoABlocksView = QuintupletsSoABlocks::View;
  using QuintupletsSoABlocksConstView = QuintupletsSoABlocks::ConstView;

  GENERATE_SOA_BLOCKS(QuintupletsLooseSoABlocksLayout,
                      SOA_BLOCK(quintupletsLoose, QuintupletsLooseSoALayout),
                      SOA_BLOCK(quintupletsOccupancy, QuintupletsOccupancySoALayout),
                      SOA_BLOCK(quintupletsByMD0, QuintupletsByMDSoALayout),
                      SOA_BLOCK(quintupletsByMD1, QuintupletsByMDSoALayout))

  using QuintupletsLooseSoABlocks = QuintupletsLooseSoABlocksLayout<>;

  // Template based accessor for getting specific SoA views. Needed in LSTEvent.dev.cc
  template <typename TSoA>
  struct QuintupletsViewAccessor;

  template <>
  struct QuintupletsViewAccessor<QuintupletsSoA> {
    static constexpr auto get(auto const& v) { return v.quintuplets(); }
  };

  template <>
  struct QuintupletsViewAccessor<QuintupletsOccupancySoA> {
    static constexpr auto get(auto const& v) { return v.quintupletsOccupancy(); }
  };

}  // namespace lst
#endif

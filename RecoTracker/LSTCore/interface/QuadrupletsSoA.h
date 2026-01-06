#ifndef RecoTracker_LSTCore_interface_QuadrupletsSoA_h
#define RecoTracker_LSTCore_interface_QuadrupletsSoA_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {
  GENERATE_SOA_LAYOUT(QuadrupletsSoALayout,
                      SOA_COLUMN(ArrayUx2,
                                 preAllocatedTripletIndices),  // pre-allocated the theoretical max triplet indices
                      SOA_COLUMN(ArrayUx2, tripletIndices),    // inner and outer triplet indices
                      SOA_COLUMN(Params_T4::ArrayU16xLayers, lowerModuleIndices),  // lower module index in each layer
                      SOA_COLUMN(Params_T4::ArrayU8xLayers, logicalLayers),        // layer ID
                      SOA_COLUMN(Params_T4::ArrayUxHits, hitIndices),              // hit indices
                      SOA_COLUMN(FPX, innerRadius),                                // inner triplet circle radius
                      SOA_COLUMN(FPX, outerRadius),                                // outer triplet radius
                      SOA_COLUMN(FPX, pt),
                      SOA_COLUMN(FPX, eta),
                      SOA_COLUMN(FPX, phi),
                      SOA_COLUMN(FPX, score_rphisum),  // r-phi based score
                      SOA_COLUMN(char, isDup),         // duplicate flag
                      SOA_COLUMN(bool, partOfTC),
                      SOA_COLUMN(float, regressionRadius),
                      SOA_COLUMN(float, nonAnchorRegressionRadius),
                      SOA_COLUMN(float, regressionCenterX),
                      SOA_COLUMN(float, regressionCenterY),
                      SOA_COLUMN(float, rzChiSquared),  // r-z only chi2
                      SOA_COLUMN(float, chiSquared),
                      SOA_COLUMN(float, nonAnchorChiSquared),
                      SOA_COLUMN(float, promptScore),
                      SOA_COLUMN(float, displacedScore),
                      SOA_COLUMN(float, fakeScore),
                      SOA_COLUMN(int, layer),
                      SOA_COLUMN(float, dBeta));

  using QuadrupletsSoA = QuadrupletsSoALayout<>;
  using Quadruplets = QuadrupletsSoA::View;
  using QuadrupletsConst = QuadrupletsSoA::ConstView;

  GENERATE_SOA_LAYOUT(QuadrupletsOccupancySoALayout,
                      SOA_COLUMN(unsigned int, nQuadruplets),
                      SOA_COLUMN(unsigned int, totOccupancyQuadruplets));

  using QuadrupletsOccupancySoA = QuadrupletsOccupancySoALayout<>;
  using QuadrupletsOccupancy = QuadrupletsOccupancySoA::View;
  using QuadrupletsOccupancyConst = QuadrupletsOccupancySoA::ConstView;

  GENERATE_SOA_BLOCKS(QuadrupletsSoABlocksLayout,
                      SOA_BLOCK(quadruplets, QuadrupletsSoALayout),
                      SOA_BLOCK(quadrupletsOccupancy, QuadrupletsOccupancySoALayout))

  using QuadrupletsSoABlocks = QuadrupletsSoABlocksLayout<>;
  using QuadrupletsSoABlocksView = QuadrupletsSoABlocks::View;
  using QuadrupletsSoABlocksConstView = QuadrupletsSoABlocks::ConstView;

  // Template based accessor for getting specific SoA views. Needed in LSTEvent.dev.cc
  template <typename TSoA>
  struct QuadrupletsViewAccessor;

  template <>
  struct QuadrupletsViewAccessor<QuadrupletsSoA> {
    static constexpr auto get(auto const& v) { return v.quadruplets(); }
  };

  template <>
  struct QuadrupletsViewAccessor<QuadrupletsOccupancySoA> {
    static constexpr auto get(auto const& v) { return v.quadrupletsOccupancy(); }
  };

}  // namespace lst
#endif

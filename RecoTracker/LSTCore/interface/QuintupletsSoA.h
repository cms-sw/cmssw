#ifndef RecoTracker_LSTCore_interface_QuintupletsSoA_h
#define RecoTracker_LSTCore_interface_QuintupletsSoA_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {
  GENERATE_SOA_LAYOUT(QuintupletsSoALayout,
                      SOA_COLUMN(ArrayUx2, tripletIndices),                        // inner and outer triplet indices
                      SOA_COLUMN(Params_T5::ArrayU16xLayers, lowerModuleIndices),  // lower module index in each layer
                      SOA_COLUMN(Params_T5::ArrayU8xLayers, logicalLayers),        // layer ID
                      SOA_COLUMN(Params_T5::ArrayUxHits, hitIndices),              // hit indices
                      SOA_COLUMN(FPX, innerRadius),                                // inner triplet circle radius
                      SOA_COLUMN(FPX, bridgeRadius),                               // "middle"/bridge triplet radius
                      SOA_COLUMN(FPX, outerRadius),                                // outer triplet radius
                      SOA_COLUMN(FPX, pt),
                      SOA_COLUMN(FPX, eta),
                      SOA_COLUMN(FPX, phi),
                      SOA_COLUMN(FPX, score_rphisum),  // r-phi based score
                      SOA_COLUMN(char, isDup),         // duplicate flag
                      SOA_COLUMN(bool, tightCutFlag),  // tight pass to be a TC
                      SOA_COLUMN(bool, partOfPT5),
                      SOA_COLUMN(float, regressionRadius),
                      SOA_COLUMN(float, regressionCenterX),
                      SOA_COLUMN(float, regressionCenterY),
                      SOA_COLUMN(float, rzChiSquared),  // r-z only chi2
                      SOA_COLUMN(float, chiSquared),
                      SOA_COLUMN(float, nonAnchorChiSquared),
                      SOA_COLUMN(float, dBeta1),
                      SOA_COLUMN(float, dBeta2));

  using QuintupletsSoA = QuintupletsSoALayout<>;
  using Quintuplets = QuintupletsSoA::View;
  using QuintupletsConst = QuintupletsSoA::ConstView;

  GENERATE_SOA_LAYOUT(QuintupletsOccupancySoALayout,
                      SOA_COLUMN(unsigned int, nQuintuplets),
                      SOA_COLUMN(unsigned int, totOccupancyQuintuplets));

  using QuintupletsOccupancySoA = QuintupletsOccupancySoALayout<>;
  using QuintupletsOccupancy = QuintupletsOccupancySoA::View;
  using QuintupletsOccupancyConst = QuintupletsOccupancySoA::ConstView;

}  // namespace lst
#endif

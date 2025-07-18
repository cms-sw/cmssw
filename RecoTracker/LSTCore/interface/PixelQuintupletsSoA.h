#ifndef RecoTracker_LSTCore_interface_PixelQuintupletsSoA_h
#define RecoTracker_LSTCore_interface_PixelQuintupletsSoA_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {
  GENERATE_SOA_LAYOUT(PixelQuintupletsSoALayout,
                      SOA_COLUMN(unsigned int, pixelSegmentIndices),
                      SOA_COLUMN(unsigned int, quintupletIndices),
                      SOA_COLUMN(Params_pT5::ArrayU16xLayers, lowerModuleIndices),  // lower module index (OT part)
                      SOA_COLUMN(Params_pT5::ArrayU8xLayers, logicalLayers),        // layer ID
                      SOA_COLUMN(Params_pT5::ArrayUxHits, hitIndices),              // hit indices
                      SOA_COLUMN(float, rPhiChiSquared),                            // chi2 from pLS to T5
                      SOA_COLUMN(float, rPhiChiSquaredInwards),                     // chi2 from T5 to pLS
                      SOA_COLUMN(float, rzChiSquared),
                      SOA_COLUMN(FPX, pixelRadius),       // pLS pt converted
                      SOA_COLUMN(FPX, quintupletRadius),  // T5 circle
                      SOA_COLUMN(FPX, eta),
                      SOA_COLUMN(FPX, phi),
                      SOA_COLUMN(FPX, score),    // used for ranking (in e.g. duplicate cleaning)
                      SOA_COLUMN(FPX, centerX),  // T3-based circle center x
                      SOA_COLUMN(FPX, centerY),  // T3-based circle center y
                      SOA_COLUMN(bool, isDup),
                      SOA_SCALAR(unsigned int, nPixelQuintuplets),
                      SOA_SCALAR(unsigned int, totOccupancyPixelQuintuplets));

  using PixelQuintupletsSoA = PixelQuintupletsSoALayout<>;
  using PixelQuintuplets = PixelQuintupletsSoA::View;
  using PixelQuintupletsConst = PixelQuintupletsSoA::ConstView;
}  // namespace lst
#endif

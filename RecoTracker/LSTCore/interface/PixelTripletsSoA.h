#ifndef RecoTracker_LSTCore_interface_PixelTripletsSoA_h
#define RecoTracker_LSTCore_interface_PixelTripletsSoA_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {
  GENERATE_SOA_LAYOUT(PixelTripletsSoALayout,
                      SOA_COLUMN(unsigned int, pixelSegmentIndices),
                      SOA_COLUMN(unsigned int, tripletIndices),
                      SOA_COLUMN(Params_pT3::ArrayU16xLayers, lowerModuleIndices),  // lower module index (OT part)
                      SOA_COLUMN(Params_pT3::ArrayU8xLayers, logicalLayers),        // layer ID
                      SOA_COLUMN(Params_pT3::ArrayUxHits, hitIndices),              // hit indices
                      SOA_COLUMN(float, rPhiChiSquared),                            // chi2 from pLS to T3
                      SOA_COLUMN(float, rPhiChiSquaredInwards),                     // chi2 from T3 to pLS
                      SOA_COLUMN(float, rzChiSquared),
                      SOA_COLUMN(FPX, pixelRadius),  // pLS pt converted
                      SOA_COLUMN(FPX, pixelRadiusError),
                      SOA_COLUMN(FPX, tripletRadius),  // T3 circle
                      SOA_COLUMN(FPX, pt),
                      SOA_COLUMN(FPX, eta),
                      SOA_COLUMN(FPX, phi),
                      SOA_COLUMN(FPX, eta_pix),  // eta from pLS
                      SOA_COLUMN(FPX, phi_pix),  // phi from pLS
                      SOA_COLUMN(FPX, score),    // used for ranking (in e.g. duplicate cleaning)
                      SOA_COLUMN(FPX, centerX),  // T3-based circle center x
                      SOA_COLUMN(FPX, centerY),  // T3-based circle center y
                      SOA_COLUMN(bool, isDup),
                      SOA_SCALAR(unsigned int, nPixelTriplets),
                      SOA_SCALAR(unsigned int, totOccupancyPixelTriplets));

  using PixelTripletsSoA = PixelTripletsSoALayout<>;
  using PixelTriplets = PixelTripletsSoA::View;
  using PixelTripletsConst = PixelTripletsSoA::ConstView;

}  // namespace lst
#endif

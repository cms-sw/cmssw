#ifndef RecoTracker_LSTCore_interface_TripletsSoA_h
#define RecoTracker_LSTCore_interface_TripletsSoA_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {
  GENERATE_SOA_LAYOUT(TripletsSoALayout,
                      SOA_COLUMN(ArrayUx2, segmentIndices),                        // inner and outer segment indices
                      SOA_COLUMN(Params_T3::ArrayU16xLayers, lowerModuleIndices),  // lower module index in each layer
                      SOA_COLUMN(Params_T3::ArrayU8xLayers, logicalLayers),        // layer ID
                      SOA_COLUMN(Params_T3::ArrayUxHits, hitIndices),              // hit indices
                      SOA_COLUMN(FPX, betaIn),     // beta/chord angle of the inner segment
                      SOA_COLUMN(float, centerX),  // lower/anchor-hit based circle center x
                      SOA_COLUMN(float, centerY),  // lower/anchor-hit based circle center y
                      SOA_COLUMN(float, radius),   // lower/anchor-hit based circle radius
#ifdef CUT_VALUE_DEBUG
                      SOA_COLUMN(float, zOut),
                      SOA_COLUMN(float, rtOut),
                      SOA_COLUMN(float, betaInCut),
#endif
                      SOA_COLUMN(bool, partOfPT5),   // is it used in a pT5
                      SOA_COLUMN(bool, partOfT5),    // is it used in a T5
                      SOA_COLUMN(bool, partOfPT3));  // is it used in a pT3

  using TripletsSoA = TripletsSoALayout<>;
  using Triplets = TripletsSoA::View;
  using TripletsConst = TripletsSoA::ConstView;

  GENERATE_SOA_LAYOUT(TripletsOccupancySoALayout,
                      SOA_COLUMN(unsigned int, nTriplets),
                      SOA_COLUMN(unsigned int, totOccupancyTriplets));

  using TripletsOccupancySoA = TripletsOccupancySoALayout<>;
  using TripletsOccupancy = TripletsOccupancySoA::View;
  using TripletsOccupancyConst = TripletsOccupancySoA::ConstView;

}  // namespace lst
#endif

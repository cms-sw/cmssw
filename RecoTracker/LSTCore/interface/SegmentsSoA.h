#ifndef RecoTracker_LSTCore_interface_SegmentsSoA_h
#define RecoTracker_LSTCore_interface_SegmentsSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
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
                      SOA_COLUMN(uint16_t, innerLowerModuleIndices),
                      SOA_COLUMN(uint16_t, outerLowerModuleIndices),
                      SOA_COLUMN(Params_LS::ArrayUxLayers, mdIndices),
                      SOA_COLUMN(unsigned int, innerMiniDoubletAnchorHitIndices),
                      SOA_COLUMN(unsigned int, outerMiniDoubletAnchorHitIndices))

  GENERATE_SOA_LAYOUT(SegmentsOccupancySoALayout,
                      SOA_COLUMN(unsigned int, nSegments),  //number of segments per inner lower module
                      SOA_COLUMN(unsigned int, totOccupancySegments))

  GENERATE_SOA_LAYOUT(SegmentsPixelSoALayout,
                      SOA_COLUMN(unsigned int, seedIdx),
                      SOA_COLUMN(int, charge),
                      SOA_COLUMN(int, superbin),
                      SOA_COLUMN(uint4, pLSHitsIdxs),
                      SOA_COLUMN(PixelType, pixelType),
                      SOA_COLUMN(char, isQuad),
                      SOA_COLUMN(char, isDup),
                      SOA_COLUMN(bool, partOfPT5),
                      SOA_COLUMN(float, ptIn),
                      SOA_COLUMN(float, ptErr),
                      SOA_COLUMN(float, px),
                      SOA_COLUMN(float, py),
                      SOA_COLUMN(float, pz),
                      SOA_COLUMN(float, etaErr),
                      SOA_COLUMN(float, eta),
                      SOA_COLUMN(float, phi),
                      SOA_COLUMN(float, score),
                      SOA_COLUMN(float, circleCenterX),
                      SOA_COLUMN(float, circleCenterY),
                      SOA_COLUMN(float, circleRadius))

  using SegmentsSoA = SegmentsSoALayout<>;
  using SegmentsOccupancySoA = SegmentsOccupancySoALayout<>;
  using SegmentsPixelSoA = SegmentsPixelSoALayout<>;

  using Segments = SegmentsSoA::View;
  using SegmentsConst = SegmentsSoA::ConstView;
  using SegmentsOccupancy = SegmentsOccupancySoA::View;
  using SegmentsOccupancyConst = SegmentsOccupancySoA::ConstView;
  using SegmentsPixel = SegmentsPixelSoA::View;
  using SegmentsPixelConst = SegmentsPixelSoA::ConstView;

}  // namespace lst

#endif

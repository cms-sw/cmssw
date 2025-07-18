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

  using SegmentsSoA = SegmentsSoALayout<>;
  using SegmentsOccupancySoA = SegmentsOccupancySoALayout<>;

  using Segments = SegmentsSoA::View;
  using SegmentsConst = SegmentsSoA::ConstView;
  using SegmentsOccupancy = SegmentsOccupancySoA::View;
  using SegmentsOccupancyConst = SegmentsOccupancySoA::ConstView;

}  // namespace lst

#endif

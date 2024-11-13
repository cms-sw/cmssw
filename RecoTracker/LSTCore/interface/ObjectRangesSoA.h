#ifndef RecoTracker_LSTCore_interface_ObjectRangesSoA_h
#define RecoTracker_LSTCore_interface_ObjectRangesSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {

  GENERATE_SOA_LAYOUT(ObjectRangesSoALayout,
                      SOA_COLUMN(ArrayIx2, mdRanges),
                      SOA_COLUMN(ArrayIx2, segmentRanges),
                      SOA_COLUMN(ArrayIx2, tripletRanges),
                      SOA_COLUMN(ArrayIx2, quintupletRanges),
                      SOA_COLUMN(int, miniDoubletModuleIndices),
                      SOA_COLUMN(int, miniDoubletModuleOccupancy),
                      SOA_COLUMN(int, segmentModuleIndices),
                      SOA_COLUMN(int, segmentModuleOccupancy),
                      SOA_COLUMN(int, tripletModuleIndices),
                      SOA_COLUMN(int, tripletModuleOccupancy),
                      SOA_COLUMN(int, quintupletModuleIndices),
                      SOA_COLUMN(int, quintupletModuleOccupancy),
                      SOA_COLUMN(uint16_t, indicesOfEligibleT5Modules),
                      SOA_SCALAR(unsigned int, nTotalMDs),
                      SOA_SCALAR(unsigned int, nTotalSegs),
                      SOA_SCALAR(unsigned int, nTotalTrips),
                      SOA_SCALAR(unsigned int, nTotalQuints),
                      SOA_SCALAR(uint16_t, nEligibleT5Modules))

  using ObjectRangesSoA = ObjectRangesSoALayout<>;

  using ObjectRanges = ObjectRangesSoA::View;
  using ObjectRangesConst = ObjectRangesSoA::ConstView;

}  // namespace lst

#endif

#ifndef L1Trigger_L1TMuon_GeometryHelpers
#define L1Trigger_L1TMuon_GeometryHelpers

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0SegmentCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"

namespace{
  // stubs
  typedef std::pair<GEMDetId, GEMPadDigi> GEMPadDigiId;
  typedef std::pair<CSCDetId, CSCCorrelatedLCTDigi> CSCCorrelatedLCTDigiId;
  typedef std::pair<DTChamberId, L1MuDTChambPhDigi> L1MuDTChambPhDigiId;
};

namespace L1TMuon
{
  namespace GeometryHelpers
  {
    // get the location of the ME0 segment in global coordinates
    GlobalPoint globalPositionOfME0LCT(const ME0Geometry* geometry,
                                       const ME0Segment& seg);

    // get the location of the GEM pad in global coordinates
    GlobalPoint globalPositionOfGEMPad(const GEMGeometry* geometry,
                                       const GEMPadDigi& gempad,
                                       const GEMDetId& id);

    // get the location of the GEM Copad in global coordinates
    GlobalPoint globalPositionOfGEMCoPad(const GEMGeometry* geometry,
                                         const GEMCoPadDigi& gempad,
                                         const GEMDetId& id);

    // get the location of the CSC LCT in global coordinates
    GlobalPoint globalPositionOfCSCLCT(const CSCGeometry* geometry,
                                       const CSCCorrelatedLCTDigi& stub,
                                       const CSCDetId& id);

  };
};

#endif

#ifndef L1Trigger_L1TMuonEndCapPhase2_SubsystemTags_h
#define L1Trigger_L1TMuonEndCapPhase2_SubsystemTags_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/L1TMuon/interface/CPPFDigi.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCluster.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigi.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"

// Forward declarations
class CSCGeometry;
class RPCGeometry;
class GEMGeometry;
class ME0Geometry;

namespace emtf::phase2 {

  struct DTTag {
    typedef L1MuDTChambPhDigi digi_type;
    typedef L1MuDTChambPhContainer collection_type;
    typedef L1MuDTChambThDigi theta_digi_type;
    typedef L1MuDTChambThContainer theta_collection_type;
  };

  struct CSCTag {
    typedef CSCDetId detid_type;
    typedef CSCCorrelatedLCTDigi digi_type;
    typedef CSCCorrelatedLCTDigiCollection collection_type;
    typedef CSCComparatorDigi comparator_digi_type;
    typedef CSCComparatorDigiCollection comparator_collection_type;
    typedef CSCGeometry detgeom_type;
  };

  struct RPCTag {
    typedef RPCDetId detid_type;
    typedef RPCDigi digi_type;
    typedef RPCDigiCollection collection_type;
    typedef RPCRecHit rechit_type;
    typedef RPCRecHitCollection rechit_collection_type;
    typedef RPCGeometry detgeom_type;
  };

  struct IRPCTag {
    typedef RPCDetId detid_type;
    typedef RPCDigi digi_type;
    typedef RPCDigiCollection collection_type;
    typedef RPCRecHit rechit_type;
    typedef RPCRecHitCollection rechit_collection_type;
    typedef RPCGeometry detgeom_type;
  };

  struct CPPFTag {
    typedef l1t::CPPFDigi digi_type;
    typedef l1t::CPPFDigiCollection collection_type;
  };

  struct GEMTag {
    typedef GEMDetId detid_type;
    typedef GEMPadDigiCluster digi_type;
    typedef GEMPadDigiClusterCollection collection_type;
    typedef GEMGeometry detgeom_type;
  };

  struct ME0Tag {
    typedef ME0DetId detid_type;
    typedef ME0TriggerDigi digi_type;
    typedef ME0TriggerDigiCollection collection_type;
    typedef ME0Geometry detgeom_type;
  };

  struct GE0Tag {
    typedef GEMDetId detid_type;
    typedef ME0TriggerDigi digi_type;
    typedef GE0TriggerDigiCollection collection_type;
    typedef GEMGeometry detgeom_type;
  };

}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_SubsystemTags_h

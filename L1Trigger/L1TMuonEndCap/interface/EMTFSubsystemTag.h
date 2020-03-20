#ifndef L1TMuonEndCap_EMTFSubsystemTag_h
#define L1TMuonEndCap_EMTFSubsystemTag_h

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/L1TMuon/interface/CPPFDigi.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCluster.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigi.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"

namespace emtf {

  struct CSCTag {
    typedef CSCCorrelatedLCTDigi digi_type;
    typedef CSCCorrelatedLCTDigiCollection digi_collection;
  };

  struct RPCTag {
    typedef RPCDigi digi_type;
    typedef RPCDigiCollection digi_collection;
  };

  struct CPPFTag {
    typedef l1t::CPPFDigi digi_type;
    typedef l1t::CPPFDigiCollection digi_collection;
  };

  struct GEMTag {
    typedef GEMPadDigiCluster digi_type;
    typedef GEMPadDigiClusterCollection digi_collection;
  };

  struct IRPCTag {
    typedef RPCDigi digi_type;
    typedef RPCDigiCollection digi_collection;
  };

  struct ME0Tag {
    typedef ME0TriggerDigi digi_type;
    typedef ME0TriggerDigiCollection digi_collection;
  };

}  //  namespace emtf

#endif

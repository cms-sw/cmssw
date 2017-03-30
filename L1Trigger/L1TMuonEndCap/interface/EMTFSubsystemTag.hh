#ifndef L1TMuonEndCap_EMTFSubsystemTag_hh
#define L1TMuonEndCap_EMTFSubsystemTag_hh

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"


namespace emtf {

  struct CSCTag {
    typedef CSCCorrelatedLCTDigi           digi_type;
    typedef CSCCorrelatedLCTDigiCollection digi_collection;
  };

  struct RPCTag {
    typedef RPCDigi           digi_type;
    typedef RPCDigiCollection digi_collection;
  };

}  //  namespace emtf

#endif

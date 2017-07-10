#ifndef EventFilter_L1TRawToDigi_Omtf_MuonUnpacker_H
#define EventFilter_L1TRawToDigi_Omtf_MuonUnpacker_H

#include <string>


#include "EventFilter/L1TRawToDigi/interface/OmtfDataWord64.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

namespace omtf { class MuonDataWord64; }

namespace omtf {

class MuonUnpacker {

public:

  void unpack(unsigned int fed, unsigned int amc, const MuonDataWord64 &raw, l1t::RegionalMuonCandBxCollection * muColl);

};

}
#endif


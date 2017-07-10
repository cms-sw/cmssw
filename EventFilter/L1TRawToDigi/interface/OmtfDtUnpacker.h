#ifndef EventFilter_L1TRawToDigi_Omtf_DtUnpacker_H
#define EventFilter_L1TRawToDigi_Omtf_DtUnpacker_H

#include <string>


#include "EventFilter/L1TRawToDigi/interface/OmtfDataWord64.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

namespace omtf {

class DtDataWord64;

class DtUnpacker {

public:

  void unpack(unsigned int fed, unsigned int amc, const DtDataWord64 &raw, std::vector<L1MuDTChambPhDigi> & phi_Container, std::vector<L1MuDTChambThDigi> & the_Container);

};

}
#endif


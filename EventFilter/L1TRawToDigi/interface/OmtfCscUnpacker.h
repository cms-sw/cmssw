#ifndef EventFilter_L1TRawToDigi_Omtf_CscUnpacker_H
#define EventFilter_L1TRawToDigi_Omtf_CscUnpacker_H

#include <string>

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfLinkMappingCsc.h"

namespace edm { class EventSetup; }

namespace omtf {

class CscDataWord64;

class CscUnpacker {

public:

  void init();
  void unpack(unsigned int fed, unsigned int amc, const CscDataWord64 &raw, CSCCorrelatedLCTDigiCollection* prod);

private:
  MapEleIndex2CscDet      theOmtf2CscDet;
};

}
#endif


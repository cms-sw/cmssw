#ifndef EventFilter_L1TRawToDigi_Omtf_CscPacker_H
#define EventFilter_L1TRawToDigi_Omtf_CscPacker_H

#include <string>

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfLinkMappingCsc.h"

namespace edm { class EventSetup; }

namespace omtf {

class CscPacker {

public:

  void init();
  void pack(const CSCCorrelatedLCTDigiCollection* prod, FedAmcRawsMap & raws);

private:
  MapCscDet2EleIndex       theCsc2Omtf;
};
}
#endif

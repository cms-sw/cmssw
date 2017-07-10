#ifndef EventFilter_L1TRawToDigi_Omtf_DtPacker_H
#define EventFilter_L1TRawToDigi_Omtf_DtPacker_H

#include <string>

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

#include "EventFilter/L1TRawToDigi/interface/OmtfDataWord64.h"

namespace omtf {

class DtPacker {

public:

  void pack(const L1MuDTChambPhContainer* phCont, const L1MuDTChambThContainer* thCont, FedAmcRawsMap & raws);

private:
};
}
#endif

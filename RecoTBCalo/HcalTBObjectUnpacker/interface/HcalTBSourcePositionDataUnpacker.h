/* -*- C++ -*- */
#ifndef HcalTBSourcePositionDataUnpacker_h_included
#define HcalTBSourcePositionDataUnpacker_h_included 1

#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

namespace hcaltb {
  class HcalTBSourcePositionDataUnpacker {
  public:
    HcalTBSourcePositionDataUnpacker(void) { }

    void unpack(const FEDRawData& raw,
		HcalSourcePositionData& hspd) const;
  };
}

#endif // HcalTBSourcePositionDataUnpacker_h_included

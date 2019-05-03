/* -*- C++ -*- */
#ifndef HcalTBSourcePositionDataUnpacker_h_included
#define HcalTBSourcePositionDataUnpacker_h_included 1

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"

namespace hcaltb {
class HcalTBSourcePositionDataUnpacker {
public:
  HcalTBSourcePositionDataUnpacker(void) {}

  void unpack(const FEDRawData &raw, HcalSourcePositionData &hspd) const;
};
} // namespace hcaltb

#endif // HcalTBSourcePositionDataUnpacker_h_included

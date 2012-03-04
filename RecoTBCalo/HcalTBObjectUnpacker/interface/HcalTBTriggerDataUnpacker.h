/* -*- C++ -*- */
#ifndef HcalTBTriggerDataUnpacker_h_included
#define HcalTBTriggerDataUnpacker_h_included 1

#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

namespace hcaltb {
  class HcalTBTriggerDataUnpacker {
  public:
    HcalTBTriggerDataUnpacker(void) { }

    void unpack(const FEDRawData& raw, HcalTBTriggerData& htbtd) const;
    static const int STANDARD_FED_ID=1;
  };
}

#endif // HcalTBTriggerDataUnpacker_h_included

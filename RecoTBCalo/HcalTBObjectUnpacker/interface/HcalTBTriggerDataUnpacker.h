/* -*- C++ -*- */
#ifndef HcalTBTriggerDataUnpacker_h_included
#define HcalTBTriggerDataUnpacker_h_included 1

#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

namespace hcaltb {
  class HcalTBTriggerDataUnpacker {
  public:
    HcalTBTriggerDataUnpacker(void) { }

    void unpack(const raw::FEDRawData& raw, hcaltb::HcalTBTriggerData& htbtd);
     
  };
}

#endif // HcalTBTriggerDataUnpacker_h_included

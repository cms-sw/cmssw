/* -*- C++ -*- */
#ifndef HcalTBSlowDataUnpacker_h_included
#define HcalTBSlowDataUnpacker_h_included 1

#include "TBDataFormats/HcalTBObjects/interface/HcalTBRunData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

namespace hcaltb {
  class HcalTBSlowDataUnpacker {
  public:
    HcalTBSlowDataUnpacker(void) { }

    void unpack(const FEDRawData&    raw,
		HcalTBRunData&            htbrd,
		HcalTBEventPosition&      htbep);
  };
}

#endif // HcalTBSlowDataUnpacker_h_included

/* -*- C++ -*- */
#ifndef HcalTBSourcePositionDataUnpacker_h_included
#define HcalTBSourcePositionDataUnpacker_h_included 1

#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "DataFormats/HcalDigi/interface/HcalUHTRhistogramDigiCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include <set>

class HcalSourcingUTCAunpacker {
public:
  HcalSourcingUTCAunpacker(void) { }

  void unpack(const FEDRawData&  raw, const HcalElectronicsMap emap, std::auto_ptr<HcalUHTRhistogramDigiCollection>&  histoDigiCollection) const;
  std::set<HcalElectronicsId> unknownIds_;
};

#endif // HcalSourcingUTCAunpacker_h_included

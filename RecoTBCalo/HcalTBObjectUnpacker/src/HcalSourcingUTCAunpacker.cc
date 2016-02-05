#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalSourcingUTCAunpacker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <string>


/// Per Event Header Structure
struct eventHeader {
  uint32_t h0;
  uint32_t h1;
  uint32_t h2;
  uint32_t h3;
};

  
void HcalSourcingUTCAunpacker::unpack(const FEDRawData&  raw, const HcalElectronicsMap emap, std::auto_ptr<std::vector<HcalHistogramDigi> > histoDigis) const {
  
  if (raw.size()<32*38) {
    throw cms::Exception("Missing Data") << "Less than 1 histogram in event";
  }

  const struct eventHeader* eh =
    (const struct eventHeader*)(raw.data());
  
//  if (raw.size()<sizeof(xdaqSourcePositionDataFormat)) {
//    throw cms::Exception("DataFormatError","Fragment too small");
//  }
//Read event header
  int numHistos  = eh->h3&0xFFFF0000;
  int numBins    = eh->h3&0x0000FFFE; //includes overflow and header word
//  bool sepCapIds = eh->h3&0x00000001;

//Set histogram word pointer to first histogram    
  uint32_t *word = (uint32_t*)(raw.data())+32*4;
  int crate   = -1;
  int slot    = -1;
  int fiber   = -1;
  int channel = -1;
  int cap     = -1;
//Loop over data
  for (int iHist = 0; iHist<numHistos; iHist++) { 
    crate   = *word&0x00FF0000;
    slot    = *word&0x0000F000;
    fiber   = *word&0x00000F80;
    channel = *word&0x0000007C;
    cap     = *word&0x00000003;

    HcalElectronicsId eid(crate, slot, fiber, channel, false);
  //  eid.setHTR(htr_cr,htr_slot,htr_tb);
    DetId did=emap.lookup(eid);
    if (did.null() || did.det()!=DetId::Hcal || did.subdetId()==0) {
      if (unknownIds_.find(eid)==unknownIds_.end()) {
        edm::LogWarning("HCAL") << "HcalHistogramUnpacker: No match found for electronics id :" << eid;
       // unknownIds_.insert(eid);
      }
      continue;
    }
    histoDigis->push_back(HcalHistogramDigi(HcalDetId(did)));
    HcalHistogramDigi& digi=histoDigis->back();
    word+=32;
    uint32_t* digiBin = digi.getArray(cap);
    for(int iBin = 0; iBin<numBins-1; iBin++) {
      digiBin = word;
      word+=32;
      digiBin+=32;
    }
    word+=32;  //skip over the overflow bin
  }
}

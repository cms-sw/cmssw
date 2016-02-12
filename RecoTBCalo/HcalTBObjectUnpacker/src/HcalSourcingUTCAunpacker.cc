#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalSourcingUTCAunpacker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <string>


/// Per Event Header Structure
struct eventHeader {
  uint32_t cdf0;
  uint32_t cdf1;
  uint32_t cdf2;
  uint32_t cdf3;
  uint32_t h0;
  uint32_t h1;
  uint32_t h2;
  uint32_t h3;
};

  
void HcalSourcingUTCAunpacker::unpack(const FEDRawData&  raw, const HcalElectronicsMap emap, std::auto_ptr<HcalUHTRhistogramDigiCollection>& histoDigiCollection) const {
  std::cout << "Unpacker Time!" << std::endl; 
  if (raw.size()<32*38) {
    throw cms::Exception("Missing Data") << "Less than 1 histogram in event";
  }
  const uint32_t* pData = (const uint32_t*) raw.data(); 
  int nwords=raw.size()/4;
  for (int iw=0; iw<nwords; iw++)
    printf("%04d %04d\n",iw,pData[iw]);
  
  const struct eventHeader* eh =
    (const struct eventHeader*)(raw.data());
  
//  if (raw.size()<sizeof(xdaqSourcePositionDataFormat)) {
//    throw cms::Exception("DataFormatError","Fragment too small");
//  }
//Read event header
  int numHistos  = ((eh->h3)>>16)&0xFFFF;
  std::cout << "Number of Histograms: " << numHistos << std::endl;
  int numBins    = ((eh->h3)>>1)&0x0000FFFE; //includes overflow and header word
  std::cout << "Bins per Histogram: " << numBins << std::endl;
  bool sepCapIds = eh->h3&0x00000001;
  std::cout << "Separate CapIds: " << sepCapIds << std::endl;

  histoDigiCollection.reset(new HcalUHTRhistogramDigiCollection(numBins, sepCapIds));
//Set histogram word pointer to first histogram    
  int crate   = -1;
  int slot    = -1;
  int fiber   = -1;
  int channel = -1;
  int cap     = -1;
//Loop over data
  pData+=4;
  for (int iHist = 0; iHist<numHistos; iHist++) {
    std::cout << "Histogram: " << iHist << std::endl;
    pData+=iHist*numBins;
    crate   = ((*pData)>>16)&0x00FF0000;
    std::cout << "Crate: " << crate << std::endl;
    slot    = ((*pData)>>12)&0x0000F000;
    std::cout << "Slot: " << slot << std::endl;
    fiber   = ((*pData)>>7)&0x00000F80;
    std::cout << "Fiber: " << fiber << std::endl;
    channel = ((*pData)>>2)&0x0000007C;
    std::cout << "Channel: " << channel << std::endl;
    cap     = *pData&0x00000003;
    std::cout << "CapId: " << cap << std::endl;
    HcalElectronicsId eid(crate, slot, fiber, channel, false);
  //  eid.setHTR(htr_cr,htr_slot,htr_tb);
    DetId did=emap.lookup(eid);
    if (did.null() || did.det()!=DetId::Hcal || did.subdetId()==0) {
      if (unknownIds_.find(eid)==unknownIds_.end()) {
        edm::LogWarning("HCAL") << "HcalHistogramUnpacker: No match found for electronics id :" << eid;
      }
      continue;
    }
    HcalUHTRhistogramDigiMutable digi = histoDigiCollection->addHistogram( did );
    for(int iBin = 0; iBin<numBins; iBin++) {
      digi.fillBin(cap, iBin, pData[iBin+1]);
      std::cout << "CapId: " << cap << "Bin: " << iBin << "Val: " << pData[iBin+1] << std::endl;
        
    }

  }
  std::cout << "DONE" << std::endl;
}

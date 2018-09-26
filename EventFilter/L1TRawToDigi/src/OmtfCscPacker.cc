#include "EventFilter/L1TRawToDigi/interface/OmtfCscPacker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1TMuon/interface/OMTF/OmtfCscDataWord64.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

namespace omtf {

void CscPacker::init()
{
  theCsc2Omtf = mapCscDet2EleIndex();
}

void CscPacker::pack(const CSCCorrelatedLCTDigiCollection* prod, FedAmcRawsMap & raws) 
{
  const CSCCorrelatedLCTDigiCollection & cscDigis = *prod;
  for (const auto & chDigis : cscDigis) {
    CSCDetId chamberId = CSCDetId(chDigis.first).chamberId();
    for (auto digi = chDigis.second.first; digi != chDigis.second.second; digi++) {
      CscDataWord64 data;
      data.hitNum_ = digi->getTrknmb();
      data.vp_ = digi->isValid();
      data.bxNum_ = digi->getBX()-(CSCConstants::LCT_CENTRAL_BX-3);
      data.halfStrip_ = digi->getStrip();
      data.clctPattern_ = digi->getPattern();
      data.keyWG_ = digi->getKeyWG();
      data.lr_ = digi->getBend();
      data.quality_ = digi->getQuality();
      auto im = theCsc2Omtf.find(chamberId);
      if (im != theCsc2Omtf.end()) {
        std::vector<EleIndex> links = {im->second.first, im->second.second};
        for (const auto & link : links) {
          unsigned int fed = link.fed();
          if (fed == 0) continue;
          data.station_ = chamberId.station()-1;
          data.linkNum_ = link.link();
          data.cscID_ = chamberId.chamber()-(link.amc()-1)*6;
          unsigned int amc =  link.amc()*2-1;
          raws[std::make_pair(fed,amc)].push_back(data.rawData);
          LogTrace("") <<"ADDED RAW: fed: "<< fed <<" amc: "<<amc <<" CSC DATA: " << data<< std::endl;
        }
      }
    }
  }

} 

}

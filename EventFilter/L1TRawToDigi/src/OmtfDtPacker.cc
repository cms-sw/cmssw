#include "EventFilter/L1TRawToDigi/interface/OmtfDtPacker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfDtDataWord64.h"

namespace omtf {

void DtPacker::pack(const L1MuDTChambPhContainer* phCont, const L1MuDTChambThContainer* thCont, FedAmcRawsMap & raws) 
{
  const L1MuDTChambPhContainer& dtphDigisBMTF = *phCont;
  const L1MuDTChambThContainer& dtthDigisBMTF = *thCont;
  for (const auto &  chDigi : *dtphDigisBMTF.getContainer() ) {
    if (abs(chDigi.whNum()) != 2) continue;
    if (chDigi.stNum() ==4) continue;
    DtDataWord64 data;
    data.st_phi_ = chDigi.phi();
    data.st_phib_ = chDigi.phiB();
    data.st_q_    = chDigi.code();
    int bxNumber = chDigi.bxNum();
    data.bxNum_ = (3+bxNumber);
    data.st_ = chDigi.stNum()-1;
    data.valid_ = 1;
    int bxCnt = (chDigi.BxCnt() >= 0 && chDigi.BxCnt() <=3) ? chDigi.BxCnt() : 0;
    data.bcnt_st_ = bxCnt;
    data.bcnt_e0_ = bxCnt;
    data.bcnt_e1_ = bxCnt;
    data.fiber_   = chDigi.Ts2Tag();
    unsigned int amc;
    unsigned int amc2=0;
    unsigned int fed = (chDigi.whNum()==-2)? 1380: 1381;
    if (chDigi.scNum()%2 !=0) {
      amc = chDigi.scNum();
      data.sector_ = 1;
    } else {
      amc = chDigi.scNum()+1;
      data.sector_ = 0;
      amc2 = (chDigi.scNum()+11)%12; // in this case data.sector_ should be 2, fixed later
    }
    LogTrace("")<<" fed: "<< fed <<" amc: "<<amc<<" DT PH DATA: " << data << std::endl;
    raws[std::make_pair(fed,amc)].push_back(data.rawData);
    if (amc2 != 0) {
      data.sector_ = 2;
      LogTrace("")<<" fed: "<< fed <<" amc: "<<amc2<<" DT PH DATA: " << data << std::endl;
      raws[std::make_pair(fed,amc2)].push_back(data.rawData);
    }
  }

  //
  //
  //
  for (const auto &  chDigi : *dtthDigisBMTF.getContainer() ) {
    if (abs(chDigi.whNum()) != 2) continue;
    if (chDigi.stNum() ==4) continue;
    DtDataWord64 data;
    int bxNumber = chDigi.bxNum();
    data.bxNum_ = (3+bxNumber);
    data.st_ = chDigi.stNum()-1;
    data.valid_ = 1;
    unsigned int amc;
    unsigned int amc2=0;
    unsigned int fed = (chDigi.whNum()==-2)? 1380: 1381;
    if (chDigi.scNum()%2 !=0) {
      amc = chDigi.scNum();
      data.sector_ = 1;
    } else {
      amc = chDigi.scNum()+1;
      data.sector_ = 0;
      amc2 = (chDigi.scNum()+11)%12; // in this case data.sector_ should be 2, fixed later
    }
    unsigned int eta = 0;
    unsigned int etaQ = 0;
    for (unsigned int ipos=0; ipos <7; ipos++) {
      if (chDigi.position(ipos) >1 ) edm::LogError("OmtfDtPacker")<<"DT TH position to ETA,  PROBLEM !!!!";
      if (chDigi.position(ipos)==1) eta |= (1 <<ipos);
      if (chDigi.quality(ipos)==1) etaQ |= (1 <<ipos);
    }
    data.eta_qbit_ = etaQ;
    data.eta_hit_  = eta;
    bool foundDigi = false;
    for (auto & raw : raws) {
      if (raw.first.first != fed) continue;
      unsigned int amcPh = raw.first.second;
      if (amc != amcPh &&  amc2 != amcPh) continue;
      auto & words = raw.second;
      for (auto & word : words) {
        if (DataWord64::dt != DataWord64::type(word)) continue;
        DtDataWord64 dataRaw(word);
        if (dataRaw.bxNum_ != data.bxNum_) continue;
        if (dataRaw.st_    != data.st_) continue;
        if (   ( amcPh == amc && dataRaw.sector_==data.sector_ )
            || ( amcPh == amc2 && 2==dataRaw.sector_           )  ) {
          foundDigi = true;
          dataRaw.eta_qbit_ =  data.eta_qbit_;
          dataRaw.eta_hit_ =  data.eta_hit_;
          word = dataRaw.rawData;
          LogTrace("")<<"AP fed: "<< fed <<" amc: "<<amc<<" DT TH DATA: " << dataRaw << std::endl;
        }
      }
    }
    if (!foundDigi) {
      LogTrace("")<<"NFD fed: "<< fed <<" amc:  "<<amc<<" DT TH DATA: " << data<< std::endl;
      raws[std::make_pair(fed,amc)].push_back(data.rawData);
      if (amc2 != 0) {
        data.sector_ = 2;
        LogTrace("")<<"NFD fed: "<< fed <<" amc2: "<<amc2<<" DT TH DATA: " << data<< std::endl;
        raws[std::make_pair(fed,amc2)].push_back(data.rawData);
      }
    }
  }



} 

}

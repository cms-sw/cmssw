#include "EventFilter/L1TRawToDigi/interface/OmtfMuonUnpacker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/L1TRawToDigi/interface/OmtfMuonDataWord64.h"


namespace omtf {

void MuonUnpacker::unpack(unsigned int fed, unsigned int amc, const MuonDataWord64 &data, l1t::RegionalMuonCandBxCollection * producedMuonDigis)
{
  LogTrace("") <<"OMTF->MUON " << std::endl;
  LogTrace("") << data << std::endl;
  l1t::tftype  overlap = (fed==1380) ? l1t::tftype::omtf_neg :  l1t::tftype::omtf_pos;
  unsigned int iProcessor = amc-1;   //0-5
  l1t::RegionalMuonCand digi;
  digi.setHwPt(data.pT());
  digi.setHwEta(data.eta());
  digi.setHwPhi(data.phi());
  digi.setHwSign(data.ch());
  digi.setHwSignValid(data.vch());
  digi.setHwQual(data.quality());
  std::map<int, int> trackAddr;
  trackAddr[0]=data.layers();
  trackAddr[1]=0;
  trackAddr[2]=data.weight_lowBits();
  digi.setTrackAddress(trackAddr);
  digi.setTFIdentifiers(iProcessor, overlap);
  int bx = data.bxNum()-3;
  LogTrace("")  <<"OMTF Muon, BX="<<bx<<", hwPt="<<digi.hwPt()<< std::endl;
  if(std::abs(bx) <= 3) producedMuonDigis->push_back(bx,digi);
}
 
}

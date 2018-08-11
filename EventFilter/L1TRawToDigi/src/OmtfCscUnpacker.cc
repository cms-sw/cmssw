#include "EventFilter/L1TRawToDigi/interface/OmtfCscUnpacker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

#include "DataFormats/L1TMuon/interface/OMTF/OmtfCscDataWord64.h"


namespace omtf {

void CscUnpacker::init()
{
  theOmtf2CscDet = mapEleIndex2CscDet();
}

void CscUnpacker::unpack(unsigned int fed, unsigned int amc, const CscDataWord64 &data, CSCCorrelatedLCTDigiCollection* prod)
{
          EleIndex omtfEle(fed, amc, data.linkNum());
          std::map<EleIndex,CSCDetId>::const_iterator icsc = theOmtf2CscDet.find(omtfEle);
          if (icsc==theOmtf2CscDet.end()) {LogTrace(" ") <<" CANNOT FIND key: " << omtfEle << std::endl; return; }
          CSCDetId cscId = theOmtf2CscDet[omtfEle];
          LogTrace("") <<"OMTF->CSC "<<cscId << std::endl;
          LogTrace("") << data << std::endl;
          if (data.linkNum() >=30) {LogTrace(" ")<<" data from overlap, skip digi "<< std::endl; return;}
          CSCCorrelatedLCTDigi digi(data.hitNum(), //trknmb
                                    data.valid(),
                                    data.quality(),
                                    data.wireGroup(),
                                    data.halfStrip(),
                                    data.clctPattern(),
                                    data.bend(),
                                    data.bxNum()+(CSCConstants::LCT_CENTRAL_BX-3) );
          LogTrace("") << digi << std::endl;
          //producedCscLctDigis->insertDigi( cscId, digi);
          prod->insertDigi( cscId, digi);

}

}

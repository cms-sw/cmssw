#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"


#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
#include "CondFormats/RPCObjects/interface/LinkConnSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include "CondFormats/RPCObjects/interface/FebConnectorSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include<iostream>


RPCReadOutMapping::RPCReadOutMapping(const std::string & version) 
  : theVersion(version) { }

const DccSpec * RPCReadOutMapping::dcc( int dccId) const
{
  IMAP im = theFeds.find(dccId);
  const DccSpec & ddc = (*im).second;
  return (im != theFeds.end()) ?  &ddc : 0;
}

void RPCReadOutMapping::add(const DccSpec & dcc)
{
  theFeds[dcc.id()] = dcc;
}


std::vector<const DccSpec*> RPCReadOutMapping::dccList() const
{
  std::vector<const DccSpec*> result;
  result.reserve(theFeds.size());
  for (IMAP im = theFeds.begin(); im != theFeds.end(); im++) {
    result.push_back( &(im->second) );
  }
  return result;
}

std::pair<int,int> RPCReadOutMapping::dccNumberRange() const
{
  
  if (theFeds.empty()) return std::make_pair(0,-1);
  else {
    IMAP first = theFeds.begin();
    IMAP last  = theFeds.end(); last--;
    return  std::make_pair(first->first, last->first);
  }
}

const LinkBoardSpec*  
    RPCReadOutMapping::location(const ChamberRawDataSpec & ele) const
{
  //FIXME after debugging change to dcc(ele.dccId)->triggerBoard(ele.dccInputChannelNum)->...
  const DccSpec *dcc = RPCReadOutMapping::dcc(ele.dccId);
  if (dcc) {
    const TriggerBoardSpec *tb = dcc->triggerBoard(ele.dccInputChannelNum);
     if (tb) {
      const LinkConnSpec *lc = tb->linkConn( ele.tbLinkInputNum);
      if (lc) {
        const LinkBoardSpec *lb = lc->linkBoard(ele.lbNumInLink);
        return lb;
      }
    }
  }
  return 0;
}


std::vector<const LinkBoardSpec*> RPCReadOutMapping::getLBforChamber(const std::string &name) const{


 std::vector<const LinkBoardSpec*> vLBforChamber;

 ChamberRawDataSpec linkboard;
 linkboard.dccId = 790;
 linkboard.dccInputChannelNum = 1;
 linkboard.tbLinkInputNum = 1;
 linkboard.lbNumInLink = 0;
 const LinkBoardSpec *location = this->location(linkboard);
 
 for(int k=0;k<18;k++){
   linkboard.dccInputChannelNum = 1;
   linkboard.tbLinkInputNum = k;
   for(int j=0;j<3;j++){
     linkboard.lbNumInLink = j;
     int febInputNum=1;
     location = this->location(linkboard);
     if (location) {
       //location->print();
       for(int j=0;j<6;j++){	 
	 const FebConnectorSpec * feb = location->feb(febInputNum+j);
	 if(feb){
	   //feb->print();	  
	   std::string chName = feb->chamber().chamberLocationName;
	   if(chName==name){
	     vLBforChamber.push_back(location);
	     //feb->chamber().print();
	     break;
	   }
	 }
       }
     }
   }
 }
 return vLBforChamber;
}

std::pair<ChamberRawDataSpec, int>  
RPCReadOutMapping::getRAWSpecForCMSChamberSrip(uint32_t  detId, int strip, int dccInputChannel) const{

 ChamberRawDataSpec linkboard;
 linkboard.dccId = 790;
 linkboard.dccInputChannelNum = dccInputChannel;

 for(int k=0;k<18;k++){
   linkboard.tbLinkInputNum = k;
   for(int j=0;j<3;j++){
     linkboard.lbNumInLink = j;
     const LinkBoardSpec *location = this->location(linkboard);    
     if (location) {
       for(int i=1;i<7;i++){	 
	 const FebConnectorSpec * feb = location->feb(i);
	 if(feb && feb->rawId()==detId){
	   for(int l=1;l<17;l++){
	     int pin = l;
	     const ChamberStripSpec *aStrip = feb->strip(pin);
	     if(aStrip && aStrip->cmsStripNumber==strip){
	       int bitInLink = (i-1)*16+l-1;
	       std::pair<ChamberRawDataSpec, int> stripInfo(linkboard,bitInLink);
	       return stripInfo;
	     }
	   }
	 }
       }
     }
   }
 }
 RPCDetId aDet(detId);
 std::cout<<"Strip: "<<strip<<" not found for detector: "<<aDet<<std::endl;
 std::pair<ChamberRawDataSpec, int> dummyStripInfo(linkboard,-99);
 return dummyStripInfo;
}



std::pair<uint32_t,int> 
RPCReadOutMapping::strip(const ChamberRawDataSpec & linkboard, int chanelLB) const{
 const LinkBoardSpec *location = this->location(linkboard);   
    const ChamberStripSpec * strip=0;

  int febInputNum = chanelLB/16+1;
  const FebConnectorSpec * feb = location->feb(febInputNum);
  if(feb){
    int pin=chanelLB%16+1;
    strip = feb->strip(pin);
    if(strip) {
      return std::make_pair(feb->rawId(),strip->cmsStripNumber);    
    }
    else{ std::cout<<"pin: "<<pin
		   <<" not found in DB."
		   <<" for LB channel number: "
		   <<chanelLB<<std::endl;  
    feb->print(2);
    edm::LogError("")<<"pin: "<<pin
		     <<" not found in DB."
		     <<" for LB channel number: "
		     <<chanelLB<<std::endl;
    }    
  }
  else{
    std::cout<<"feb: "<<febInputNum
	     <<" not found in DB."
	     <<" for LB channel number: "
	     <<chanelLB<<std::endl;  
    
  }
  return std::make_pair((uint32_t)0,(int)0);
}




RPCReadOutMapping::StripInDetUnit 
    RPCReadOutMapping::detUnitFrame(const LinkBoardSpec* location, 
    int febInLB, int stripPinInFeb) const 
{
  uint32_t detUnit = 0;
  int stripInDU = 0;

  const FebConnectorSpec * feb = location->feb(febInLB);
  if (feb) {
    detUnit = feb->rawId();
    const ChamberStripSpec * strip = feb->strip(stripPinInFeb);
    if (strip) {
      stripInDU = strip->cmsStripNumber;
    }
  }
  return std::make_pair(detUnit,stripInDU);
}



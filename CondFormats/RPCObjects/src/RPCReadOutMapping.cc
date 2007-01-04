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

std::pair< ChamberRawDataSpec, LinkBoardChannelCoding> 
    RPCReadOutMapping::rawDataFrame( uint32_t rawDetId, int stripInDU) const 
{
  ChamberRawDataSpec eleIndex = { 0,0,0,0 };

  for (IMAP im=theFeds.begin(); im != theFeds.end(); im++) {
    const DccSpec & dccSpec = (*im).second; 
//    LogTrace("rawDataFrame")<< dccSpec.print(1);
//    if (dccSpec.id() != 790) continue;
    const std::vector<TriggerBoardSpec> & triggerBoards = dccSpec.triggerBoards();
    for ( std::vector<TriggerBoardSpec>::const_iterator 
        it = triggerBoards.begin(); it != triggerBoards.end(); it++) {
      const TriggerBoardSpec & triggerBoard = (*it);
//      if (triggerBoard.dccInputChannelNum() != 13) continue;
//      LogTrace("rawDataFrame")<< triggerBoard.print(1);
      const std::vector<LinkConnSpec> & linkConns = triggerBoard.linkConns();
      for ( std::vector<LinkConnSpec>::const_iterator
          ic = linkConns.begin(); ic != linkConns.end(); ic++) {
              
        const LinkConnSpec & link = (*ic); 
//        if (link.triggerBoardInputNumber() != 17) continue;
//        LogTrace("rawDataFrame")<< link.print(1);
        const std::vector<LinkBoardSpec> & boards = link.linkBoards();
        for ( std::vector<LinkBoardSpec>::const_iterator
            ib = boards.begin(); ib != boards.end(); ib++) { 

          const LinkBoardSpec & board = (*ib);
//          if (board.linkBoardNumInLink() != 2) continue;
//          LogTrace("rawDataFrame")<< board.print(2);
          
          eleIndex.dccId = dccSpec.id();
          eleIndex.dccInputChannelNum = triggerBoard.dccInputChannelNum();
          eleIndex.tbLinkInputNum = link.triggerBoardInputNumber(); 
          eleIndex.lbNumInLink = board.linkBoardNumInLink();

          const std::vector<FebConnectorSpec> & febs = board.febs();
          int fedCheck = 0;
          for ( std::vector<FebConnectorSpec>::const_iterator
              ifc = febs.begin(); ifc != febs.end(); ifc++) {
            const FebConnectorSpec & febConnector = (*ifc);
            fedCheck++;
            if (febConnector.rawId() != rawDetId) continue;
            int fedInLB = febConnector.linkBoardInputNum();
            if (fedInLB != fedCheck) {
              edm::LogError("rawDataFrame") << " problem with fedInLB: " <<fedInLB<<" "<<fedCheck;
            }
            const std::vector<ChamberStripSpec> & strips = febConnector.strips();

//          int ipinCheck = 0;
            for (std::vector<ChamberStripSpec>::const_iterator 
                is = strips.begin(); is != strips.end(); is++) {
              const ChamberStripSpec & strip = (*is);
              int stripPinInFeb = strip.cablePinNumber;
//            if (stripPinInFeb != ++ipinCheck) {
//              edm::LogError("rawDataFrame") << " problem!, strip pin check, ipin= "<<ipinCheck
//                   <<" strip: "<< strip.print(1); 
//            }
              if ( strip.cmsStripNumber == stripInDU) {
//                std::cout << " HERE !!!! " << std::endl;
                return std::make_pair( eleIndex, LinkBoardChannelCoding( fedInLB, stripPinInFeb) ); 
              }
            } 
          } 
        }
      }
    }
  }
  std::cout << " not found in the map! " << std::endl;
  return std::make_pair( eleIndex, LinkBoardChannelCoding(0,0) );
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


RPCReadOutMapping::StripInDetUnit
    RPCReadOutMapping::strip(const ChamberRawDataSpec & linkboard, int chanelLB) const 
{
  const LinkBoardSpec *location = this->location(linkboard);
  LinkBoardChannelCoding channel(chanelLB);
  return detUnitFrame(location, channel.fedInLB(), channel.stripPinInFeb());
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
    } else {
      edm::LogError("detUnitFrame")<<"problem with stip for febInLB: "<<febInLB
                                   <<" strip pin: "<< stripPinInFeb
                                   <<" strip pin: "<< stripPinInFeb
                                   <<" for linkBoard: "<<location->print(3);
    }
  } else {
    edm::LogError("detUnitFrame")<<"problem with detUnit for febInLB: "<<febInLB
                                 <<" for linkBoard: "<<location->print(3);
  }
  return std::make_pair(detUnit,stripInDU);
}



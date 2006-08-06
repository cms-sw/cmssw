#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
#include "CondFormats/RPCObjects/interface/LinkConnSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include "CondFormats/RPCObjects/interface/FebConnectorSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"
#include "FWCore/Utilities/interface/Exception.h"

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
    else std::cout << "NO STRIP!" << std::endl;
  }
  else std::cout <<"NO FEB!!" << std::endl;
  return std::make_pair(detUnit,stripInDU);
}

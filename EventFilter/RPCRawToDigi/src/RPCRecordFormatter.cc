/** \file
 * Implementation of class RPCRecordFormatter
 *
 *  $Date: 2007/03/28 22:35:27 $
 *  $Revision: 1.27 $
 *
 * \author Ilaria Segoni
 */

#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"
#include "EventFilter/RPCRawToDigi/interface/RPCLinkBoardData.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"

#include <vector>
#include <bitset>
#include <sstream>

using namespace std;
using namespace edm;
using namespace rpcrawtodigi;


RPCRecordFormatter::RPCRecordFormatter(int fedId, const RPCReadOutMapping *r)
 : currentFED(fedId), readoutMapping(r)
{ }

RPCRecordFormatter::~RPCRecordFormatter()
{ }


std::vector<EventRecords> RPCRecordFormatter::recordPack(
    uint32_t rawDetId, const RPCDigi & digi, int trigger_BX) const 
{
  std::vector<EventRecords> result;
   
  LogTrace("") << " DIGI;  det: " << rawDetId<<", strip: "<<digi.strip()<<", bx: "<<digi.bx();
  int stripInDU = digi.strip();

  // decode digi<->map
  typedef std::vector< std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> > RawDataFrames;
  RPCReadOutMapping::StripInDetUnit duFrame(rawDetId, stripInDU);
  RawDataFrames rawDataFrames = readoutMapping->rawDataFrame(duFrame);

  for (RawDataFrames::const_iterator ir = rawDataFrames.begin(); ir != rawDataFrames.end(); ir++) {
    
    const LinkBoardElectronicIndex & eleIndex = (*ir).first;
    const LinkBoardPackedStrip & lbPackedStrip = (*ir).second;

    if (eleIndex.dccId == currentFED) {

      LogTrace("pack:")
           <<" dccId= "<<eleIndex.dccId
           <<" dccInputChannelNum= "<<eleIndex.dccInputChannelNum
           <<" tbLinkInputNum= "<<eleIndex.tbLinkInputNum
           <<" lbNumInLink="<<eleIndex.lbNumInLink;

      // BX 
      int current_BX = trigger_BX+digi.bx();
      BXRecord bxr(current_BX);

      // TB 
      int tbLinkInputNumber = eleIndex.tbLinkInputNum;
      int rmb = eleIndex.dccInputChannelNum; 
      TBRecord tbr( tbLinkInputNumber, rmb);   

      // LB record
      RPCLinkBoardData lbData;
      lbData.setLbNumber(eleIndex.lbNumInLink);
      lbData.setEod(0);
      lbData.setHalfP(0);
      int packedStrip = lbPackedStrip.packedStrip();
      vector<int> bitsOn; bitsOn.push_back(packedStrip);                        
      lbData.setPartitionNumber( packedStrip/8 );
      lbData.setBits(bitsOn);
      LBRecord lbr(lbData);

      result.push_back(  EventRecords(trigger_BX, bxr, tbr, lbr) );
    }
  }
  return result;
}

void RPCRecordFormatter::recordUnpack(
    const EventRecords & event, std::auto_ptr<RPCDigiCollection> & prod)
{
  int triggerBX = event.triggerBx();
  int currentBX = event.bxRecord().bx();
  int currentRMB = event.tbRecord().rmb(); 
  int currentTbLinkInputNumber = event.tbRecord().tbLinkInputNumber();
  RPCLinkBoardData lbData = event.lbRecord().lbData();

  LinkBoardElectronicIndex eleIndex;
  eleIndex.dccId = currentFED;
  eleIndex.dccInputChannelNum = currentRMB;
  eleIndex.tbLinkInputNum = currentTbLinkInputNumber;
  eleIndex.lbNumInLink = lbData.lbNumber();

  const LinkBoardSpec* linkBoard = readoutMapping->location(eleIndex);

  if (!linkBoard) {
    throw cms::Exception("Invalid Linkboard location!") 
              << "dccId: "<<eleIndex.dccId
              << "dccInputChannelNum: " <<eleIndex.dccInputChannelNum
              << " tbLinkInputNum: "<<eleIndex.tbLinkInputNum
              << " lbNumInLink: "<<eleIndex.lbNumInLink;
  }

  std::vector<int> bits=lbData.bitsOn();
  for(std::vector<int>::iterator pBit = bits.begin(); pBit != bits.end(); ++pBit){

    LinkBoardPackedStrip lbBit(*pBit);
    RPCReadOutMapping::StripInDetUnit duFrame = readoutMapping->detUnitFrame(*linkBoard,lbBit);

    uint32_t rawDetId = duFrame.first;
    int geomStrip = duFrame.second;
    if (!rawDetId) {
      LogError("recordUnpack: problem with rawDetId, skip LB data");
      continue;
    }

    // Creating RPC digi
    RPCDigi digi(geomStrip,currentBX-triggerBX);

    /// Committing digi to the product
    LogTrace("")<<" DIGI;  det: "<<rawDetId<<", strip: "<<digi.strip()<<", bx: "<<digi.bx();
    LogTrace("") << " LinkBoardElectronicIndex: " << eleIndex.print(); 
    prod->insertDigi(RPCDetId(rawDetId),digi);
  }
}

/** \file
 * Implementation of class RPCRecordFormatter
 *
 *  $Date: 2008/03/27 13:51:05 $
 *  $Revision: 1.33 $
 *
 * \author Ilaria Segoni
 */

#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"


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
      RecordBX bxr(current_BX);

      // LB 
      int tbLinkInputNumber = eleIndex.tbLinkInputNum;
      int rmb = eleIndex.dccInputChannelNum; 
      RecordSLD lbr( tbLinkInputNumber, rmb);   

      // CD record
      int lbInLink = eleIndex.lbNumInLink;
      int eod = 0;
      int halfP = 0;
      int packedStrip = lbPackedStrip.packedStrip();     
      int partitionNumber = packedStrip/8; 
      RecordCD cdr(lbInLink, partitionNumber, eod, halfP, vector<int>(1,packedStrip) );

      result.push_back(  EventRecords(trigger_BX, bxr, lbr, cdr) );
    }
  }
  return result;
}

int RPCRecordFormatter::recordUnpack(
    const EventRecords & event, std::auto_ptr<RPCDigiCollection> & prod)
{
  int status = 0;
  int triggerBX = event.triggerBx();
  int currentBX = event.recordBX().bx();
  int currentRMB = event.recordSLD().rmb(); 
  int currentTbLinkInputNumber = event.recordSLD().tbLinkInputNumber();

  LinkBoardElectronicIndex eleIndex;
  eleIndex.dccId = currentFED;
  eleIndex.dccInputChannelNum = currentRMB;
  eleIndex.tbLinkInputNum = currentTbLinkInputNumber;
  eleIndex.lbNumInLink = event.recordCD().lbInLink();

  if(readoutMapping == 0) return status;
  const LinkBoardSpec* linkBoard = readoutMapping->location(eleIndex);
  if (!linkBoard) {
    LogError(" ** PROBLEM ** Invalid Linkboard location, skip CD event, ") 
              << "dccId: "<<eleIndex.dccId
              << "dccInputChannelNum: " <<eleIndex.dccInputChannelNum
              << " tbLinkInputNum: "<<eleIndex.tbLinkInputNum
              << " lbNumInLink: "<<eleIndex.lbNumInLink;
    status = 1;
    return status;
  }

  std::vector<int> packStrips = event.recordCD().packedStrips();
  for(std::vector<int>::iterator is = packStrips.begin(); is != packStrips.end(); ++is) {

    RPCReadOutMapping::StripInDetUnit duFrame = 
        readoutMapping->detUnitFrame(*linkBoard, LinkBoardPackedStrip(*is) );

    uint32_t rawDetId = duFrame.first;
    int geomStrip = duFrame.second;
    if (!rawDetId) {
      LogError("** PROBLEM ** no rawDetId, skip at lease part of CD data");
      status = 2;
      continue;
    }

    // Creating RPC digi
    RPCDigi digi(geomStrip,currentBX-triggerBX);

    /// Committing digi to the product
    LogTrace("") << " LinkBoardElectronicIndex: " << eleIndex.print(); 
    LogTrace("")<<" DIGI;  det: "<<rawDetId<<", strip: "<<digi.strip()<<", bx: "<<digi.bx();
    prod->insertDigi(RPCDetId(rawDetId),digi);
  }
  return status;
}

#ifndef EventFilter_L1TRawToDigi_Omtf_RpcLinkMap_H
#define EventFilter_L1TRawToDigi_Omtf_RpcLinkMap_H

#include <fstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/L1TRawToDigi/interface/OmtfEleIndex.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "CondFormats/RPCObjects/interface/DccSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"
#include "CondFormats/RPCObjects/interface/LinkBoardPackedStrip.h"
#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"

#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"
#include "EventFilter/RPCRawToDigi/interface/RPCPackingModule.h"
#include "DataFormats/RPCDigi/interface/DataRecord.h"
#include "DataFormats/RPCDigi/interface/ReadoutError.h"
#include "DataFormats/RPCDigi/interface/RPCRawSynchro.h"
#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"
#include "EventFilter/RPCRawToDigi/interface/DebugDigisPrintout.h"
#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"

#include "CondFormats/DataRecord/interface/RPCOMTFLinkMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCAMCLinkMap.h"
#include "CondFormats/RPCObjects/interface/RPCAMCLink.h"
#include "CondFormats/RPCObjects/interface/RPCLBLink.h"

#include "DataFormats/RPCDigi/interface/DataRecord.h"
#include "DataFormats/RPCDigi/interface/RecordBX.h"
#include "DataFormats/RPCDigi/interface/RecordSLD.h"
#include "DataFormats/RPCDigi/interface/RecordCD.h"



namespace Omtf {

class RpcLinkMap {
public:
  RpcLinkMap() { }

  void init(const RPCAMCLinkMap::map_type & amcMap) {

    for (const auto & item : amcMap ) {
      unsigned int fedId = item.first.getFED();
      unsigned int amcSlot = item.first.getAMCNumber();
      unsigned int link = item.first.getAMCInput();
      std::string lbName =  item.second.getName();

      std::ostringstream processorNameStr; processorNameStr<<"OMTF";;
      if (fedId==1380) processorNameStr<< "n"; else processorNameStr<< "p";
      processorNameStr<< amcSlot/2+1;
      std::string processorName(processorNameStr.str());

      std::map< unsigned int, std::string > & li2lb = link2lbName[processorName];
      std::map< std::string, unsigned int > & lb2li = lbName2link[processorName];
      li2lb[link] = lbName;
      lb2li[lbName] = link;
      EleIndex ele(processorName, link);
      lbName2OmtfIndex[lbName].push_back(ele);
    }
  }

  void init( const std::string& fName) {
    std::ifstream inFile;
    inFile.open(fName);
    if (inFile) {
      LogTrace("")<<" reading OmtfRpcLinksMap from: "<<fName;
    } else {
      LogTrace("")<<" Unable to open file "<<fName;

      throw std::runtime_error("Unable to open OmtfRpcLinksMap file " + fName);
    }

    std::string line;
    while (std::getline(inFile, line)) {
      line.erase(0, line.find_first_not_of(" \t\r\n"));      //cut first character
      if (line.empty() || !line.compare(0,2,"--")) continue; // empty or comment line
      std::stringstream ss(line);
      std::string processorName, lbName;
      unsigned int link, dbId;
      if (ss >> processorName >> link >> lbName >> dbId) {
          std::map< unsigned int, std::string > & li2lb = link2lbName[processorName];
          std::map< std::string, unsigned int > & lb2li = lbName2link[processorName];
          li2lb[link] = lbName;
          lb2li[lbName] = link;
         EleIndex ele(processorName, link);
          lbName2OmtfIndex[lbName].push_back(ele);
      }
    }
    inFile.close();
  }

  const std::string & lbName(const std::string& board, unsigned int link) const {
    return link2lbName.at(board).at(link);
  }

  unsigned int        link(const std::string& board, const std::string& lbName) const {
    return lbName2link.at(board).at(lbName);
  }

  const std::vector<EleIndex> &omtfEleIndex ( const std::string& lbName) const {
     return lbName2OmtfIndex.at(lbName);
  }
private:
    std::map<std::string, std::map<unsigned int, std::string> > link2lbName; //[processorName][rpcRxNum] - lbName
    std::map<std::string, std::map<std::string, unsigned int> > lbName2link; //[processorName][lbName] - rpcRxNum
    std::map<std::string, std::vector<EleIndex> > lbName2OmtfIndex; //[lbName] - vector of {board,rpcRxNum}

};

}; // namespace Omtf
#endif

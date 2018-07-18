#include "EventFilter/L1TRawToDigi/interface/OmtfLinkMappingRpc.h"

#include <fstream>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "CondFormats/RPCObjects/interface/DccSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"
#include "CondFormats/RPCObjects/interface/LinkBoardPackedStrip.h"
#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"

#include "CondFormats/DataRecord/interface/RPCOMTFLinkMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCAMCLinkMap.h"

namespace omtf {

bool lessLinkBoardElectronicIndex::operator() (const LinkBoardElectronicIndex & o1, const LinkBoardElectronicIndex & o2) const {
  if (o1.dccId < o2.dccId) return true;
  else if (o1.dccId == o2.dccId && o1.dccInputChannelNum < o2.dccInputChannelNum) return true;
  else if (o1.dccId == o2.dccId && o1.dccInputChannelNum == o2.dccInputChannelNum && o1.tbLinkInputNum < o2.tbLinkInputNum) return true;
  else return false;
}


MapEleIndex2LBIndex translateOmtf2Pact(const RpcLinkMap & omtfLink2Ele, const RPCReadOutMapping* pactCabling) {
  MapEleIndex2LBIndex omtf2rpc;

  std::vector<const DccSpec*> dccs = pactCabling->dccList();
  for (auto it1 : dccs) {
    const std::vector<TriggerBoardSpec> & rmbs = it1->triggerBoards();
    for (auto const & it2 : rmbs) {
      const  std::vector<LinkConnSpec> & links = it2.linkConns();
      for (auto const & it3 : links) {
        const  std::vector<LinkBoardSpec> & lbs = it3.linkBoards();
        for (std::vector<LinkBoardSpec>::const_iterator it4=lbs.begin(); it4 != lbs.end(); ++it4) {

          std::string lbNameCH = it4->linkBoardName();
          std::string lbName = lbNameCH.substr(0,lbNameCH.size()-4);
          std::vector<EleIndex> omtfEles = omtfLink2Ele.omtfEleIndex(lbName);
//        if (!omtfEles.empty()) std::cout <<"  isOK ! " <<  it4->linkBoardName() <<" has: " << omtfEles.size() << " first: "<< omtfEles[0] << std::endl;
          LinkBoardElectronicIndex rpcEle = { it1->id(), it2.dccInputChannelNum(), it3.triggerBoardInputNumber(), it4->linkBoardNumInLink()};
          for ( const auto & omtfEle : omtfEles ) omtf2rpc[omtfEle]= rpcEle;
        }
      }
    }
  }
  LogTrace(" ") << " SIZE OF OMTF to RPC map TRANSLATION is: " << omtf2rpc.size() << std::endl;
  return omtf2rpc;
}

MapLBIndex2EleIndex translatePact2Omtf(const RpcLinkMap & omtfLink2Ele, const RPCReadOutMapping* pactCabling) {
  MapLBIndex2EleIndex pact2omtfs;
  MapEleIndex2LBIndex omtf2rpcs = translateOmtf2Pact(omtfLink2Ele, pactCabling);
  for ( const auto & omtf2rpc : omtf2rpcs) {
    std::pair<EleIndex,EleIndex> & omtfs = pact2omtfs[omtf2rpc.second];
    if (omtfs.first.fed()==0) omtfs.first = omtf2rpc.first;
    else if (omtfs.second.fed()==0) omtfs.second = omtf2rpc.first; 
    else edm::LogError(" translatePact2Omtf ") << " PROBLEM LinkBoardElectronicIndex already USED!!!! ";
  }
  return pact2omtfs;
}

void RpcLinkMap::init(const edm::EventSetup& es) {

  edm::ESHandle<RPCAMCLinkMap> amcMapping;
  es.get<RPCOMTFLinkMapRcd>().get(amcMapping);
  const RPCAMCLinkMap::map_type & amcMap = amcMapping->getMap();

  for (const auto & item : amcMap ) {
    unsigned int fedId = item.first.getFED();
    unsigned int amcSlot = item.first.getAMCNumber();
    unsigned int link = item.first.getAMCInput();
    std::string lbName =  item.second.getName();

    std::string processorNameStr = "OMTF";;
    if (fedId==1380) processorNameStr = "OMTFn"; else processorNameStr = "OMTFp";
    processorNameStr += std::to_string(amcSlot/2+1);
    std::string processorName = processorNameStr;

    std::map< unsigned int, std::string > & li2lb = link2lbName[processorName];
    std::map< std::string, unsigned int > & lb2li = lbName2link[processorName];
    li2lb[link] = lbName;
    lb2li[lbName] = link;
    EleIndex ele(processorName, link);
    lbName2OmtfIndex[lbName].push_back(ele);
  }
}

void RpcLinkMap::init(const std::string& fName) {
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

std::vector<EleIndex> RpcLinkMap::omtfEleIndex ( const std::string& lbName) const { 
    const auto pos = lbName2OmtfIndex.find(lbName);
    return pos != lbName2OmtfIndex.end() ?  pos->second : std::vector<EleIndex>() ; 
}


}

#include "EventFilter/L1TRawToDigi/interface/OmtfRpcPacker.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"
#include "EventFilter/RPCRawToDigi/interface/RPCPackingModule.h"
#include "EventFilter/RPCRawToDigi/interface/DebugDigisPrintout.h"
#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfRpcDataWord64.h"

namespace omtf {

void RpcPacker::initCabling(const edm::EventSetup & es) {
  edm::ESTransientHandle<RPCEMap> readoutMapping;
  es.get<RPCEMapRcd>().get(readoutMapping);
  thePactCabling.reset(readoutMapping->convert());
  LogDebug("OmtfPacker") <<" Has PACT readout map, VERSION: " << thePactCabling->version() << std::endl;
}

void RpcPacker::init(const edm::EventSetup & es)
{
  initCabling(es);
  RpcLinkMap omtfLink2Ele;
  omtfLink2Ele.init(es);
  thePact2Omtf = translatePact2Omtf(omtfLink2Ele,thePactCabling.get());
}

void RpcPacker::init(const edm::EventSetup & es, const std::string & connectionFile)
{
  initCabling(es);
  RpcLinkMap omtfLink2Ele;
  omtfLink2Ele.init(connectionFile);
  thePact2Omtf = translatePact2Omtf(omtfLink2Ele,thePactCabling.get());
}

void RpcPacker::pack(const RPCDigiCollection * digis, FedAmcRawsMap & raws) 
{
  LogTrace("")<<" HERE HERE !!! RPC PACKER" << rpcrawtodigi::DebugDigisPrintout()(digis);
  for (int dcc=790; dcc <= 792; dcc++) {
    RPCRecordFormatter formatter(dcc, thePactCabling.get());
    const std::vector<rpcrawtodigi::EventRecords> & merged = RPCPackingModule::eventRecords(dcc,200, digis ,formatter);
    LogTrace("") << " SIZE OF MERGED, for DCC="<<dcc<<" is: "<<merged.size()<<std::endl;
    for (const auto & rpcEvent : merged) {
      RpcDataWord64 data;
      data.bxNum_ =  rpcEvent.dataToTriggerDelay();
      data.frame1_ = rpcEvent.recordCD().data();
      LinkBoardElectronicIndex rpcEle = { dcc, rpcEvent.recordSLD().rmb(),  rpcEvent.recordSLD().tbLinkInputNumber(), rpcEvent.recordCD().lbInLink() };
      auto it = thePact2Omtf.find(rpcEle);
      if (it != thePact2Omtf.end()) {
        const EleIndex & omtfEle1 = it->second.first;
        const EleIndex & omtfEle2 = it->second.second;
        if(omtfEle1.fed()) { data.linkNum_ = omtfEle1.link(); raws[std::make_pair(omtfEle1.fed(), omtfEle1.amc()*2-1)].push_back(data.rawData); }
        if(omtfEle2.fed()) { data.linkNum_ = omtfEle2.link(); raws[std::make_pair(omtfEle2.fed(), omtfEle2.amc()*2-1)].push_back(data.rawData); }
      }
    }
  }

} 

};

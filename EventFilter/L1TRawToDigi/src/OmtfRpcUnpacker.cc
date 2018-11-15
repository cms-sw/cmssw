#include "EventFilter/L1TRawToDigi/interface/OmtfRpcUnpacker.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"
#include "EventFilter/RPCRawToDigi/interface/RPCPackingModule.h"
#include "EventFilter/RPCRawToDigi/interface/DebugDigisPrintout.h"
#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfRpcDataWord64.h"

#include "DataFormats/RPCDigi/interface/DataRecord.h"
#include "DataFormats/RPCDigi/interface/RecordBX.h"
#include "DataFormats/RPCDigi/interface/RecordSLD.h"
#include "DataFormats/RPCDigi/interface/RecordCD.h"

namespace omtf {

void RpcUnpacker::initCabling(const edm::EventSetup & es) 
{
  edm::ESTransientHandle<RPCEMap> readoutMapping;
  es.get<RPCEMapRcd>().get(readoutMapping);
  thePactCabling.reset(readoutMapping->convert());

  LogDebug("OmtfUnpacker") <<" Has PACT readout map, VERSION: " << thePactCabling->version() << std::endl;
}

void RpcUnpacker::init(const edm::EventSetup & es)
{
  initCabling(es);
  RpcLinkMap omtfLink2Ele;
  omtfLink2Ele.init(es);
  theOmtf2Pact = translateOmtf2Pact(omtfLink2Ele,thePactCabling.get());
}

void RpcUnpacker::init(const edm::EventSetup & es, const std::string & connectionFile)
{
  initCabling(es);
  RpcLinkMap omtfLink2Ele;
  omtfLink2Ele.init(connectionFile);
  theOmtf2Pact = translateOmtf2Pact(omtfLink2Ele,thePactCabling.get());
}

void RpcUnpacker::unpack(int triggerBX, unsigned int fed, unsigned int amc, const RpcDataWord64 &data, RPCDigiCollection * prod)
{
  LogTrace("RpcUnpacker:") << data;

//  EleIndex omtfEle(fedHeader.sourceID(), bh.getAMCNumber()/2+1, data.linkNum());
  EleIndex omtfEle(fed, amc, data.linkNum());
  LinkBoardElectronicIndex rpcEle = theOmtf2Pact.at(omtfEle);
  RPCRecordFormatter formater(rpcEle.dccId, thePactCabling.get());


  rpcrawtodigi::EventRecords records(triggerBX);
  rpcrawtodigi::RecordBX recordBX(triggerBX+data.bxNum()-3);
  records.add(recordBX);   // warning: event records must be added in right order
  rpcrawtodigi::RecordSLD recordSLD(rpcEle.tbLinkInputNum, rpcEle.dccInputChannelNum);
  records.add(recordSLD); // warning: event records must be added in right order

  for (unsigned int iframe=1; iframe <=3; iframe++) {

    uint16_t frame = (iframe==1) ?  data.frame1() : ( (iframe==2) ?  data.frame2() : data.frame3() );
    if (frame==0) continue;
    rpcrawtodigi::RecordCD recordCD(frame);
    records.add(recordCD);

    LogTrace("") <<"OMTF->RPC Event isComplete: "<<records.complete() <<records.print(rpcrawtodigi::DataRecord::StartOfBXData); // << std::endl;
    LogTrace("") <<"OMTF->RPC Event:             "<<records.print(rpcrawtodigi::DataRecord::StartOfTbLinkInputNumberData) << std::endl;
    LogTrace("") <<"OMTF->RPC Event:             "<<records.print(rpcrawtodigi::DataRecord::ChamberData)
                 <<" lb:"<< recordCD.lbInLink()
                 <<" part: "<< recordCD.partitionNumber()
                 <<" partData: "<<recordCD.partitionData()
                 << std::endl << std::endl;

    if (records.complete()) formater.recordUnpack( records, prod, nullptr,nullptr);
  }
}
}

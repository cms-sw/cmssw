#include <iostream>
#include <iomanip>
#include <bitset>
#include <string>
#include <map>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "EventFilter/L1TRawToDigi/interface/AMC13Spec.h"
#include "EventFilter/L1TRawToDigi/interface/AMCSpec.h"
#include "FWCore/Utilities/interface/CRC16.h"

#include "EventFilter/L1TRawToDigi/interface/OmtfDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfCscDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfDtDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfRpcDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfMuonDataWord64.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "EventFilter/L1TRawToDigi/interface/OmtfRpcPacker.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfCscPacker.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfDtPacker.h"

namespace omtf {

class OmtfPacker: public edm::stream::EDProducer<> {
public:

  OmtfPacker(const edm::ParameterSet& pset);

  ~OmtfPacker() {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event & ev, const edm::EventSetup& es) override;

  void beginRun(const edm::Run &run, const edm::EventSetup& es) override;

private:

  edm::ParameterSet theConfig;
  unsigned long theEventCounter;

  std::string theOutputTag;

  bool theSkipRpc, theSkipCsc, theSkipDt;

  edm::EDGetTokenT<RPCDigiCollection> theRpcToken;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> theCscToken;
  edm::EDGetTokenT<L1MuDTChambThContainer> theDtThToken;
  edm::EDGetTokenT<L1MuDTChambPhContainer> theDtPhToken;

  CscPacker theCscPacker;
  RpcPacker theRpcPacker;
  DtPacker  theDtPacker;

};

OmtfPacker::OmtfPacker(const edm::ParameterSet& pset) : theConfig(pset), theEventCounter(0) 
{
  theOutputTag = pset.getParameter<std::string>("outputTag");

  produces<FEDRawDataCollection>(theOutputTag);

  theRpcToken = consumes<RPCDigiCollection>(pset.getParameter<edm::InputTag>("rpcInputLabel"));
  theCscToken = consumes<CSCCorrelatedLCTDigiCollection>(pset.getParameter<edm::InputTag>("cscInputLabel"));
  theDtPhToken = consumes<L1MuDTChambPhContainer>(pset.getParameter<edm::InputTag>("dtPhInputLabel"));
  theDtThToken = consumes<L1MuDTChambThContainer>(pset.getParameter<edm::InputTag>("dtThInputLabel"));

  theSkipDt   = pset.getParameter<bool>("skipDt");
  theSkipRpc  = pset.getParameter<bool>("skipRpc");
  theSkipCsc  = pset.getParameter<bool>("skipCsc");

}

void OmtfPacker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("rpcInputLabel",edm::InputTag(""));
  desc.add<edm::InputTag>("cscInputLabel",edm::InputTag(""));
  desc.add<edm::InputTag>("dtPhInputLabel",edm::InputTag(""));
  desc.add<edm::InputTag>("dtThInputLabel",edm::InputTag(""));
  desc.add<bool>("skipRpc",false);
  desc.add<bool>("skipCsc",false);
  desc.add<bool>("skipDt",false);
  desc.add<bool>("useRpcConnectionFile",false);
  desc.add<std::string>("rpcConnectionFile","");
  desc.add<std::string>("outputTag","");
  descriptions.add("omtfPacker",desc);
}

void OmtfPacker::beginRun(const edm::Run &run, const edm::EventSetup& es) {

  //
  // initialise RPC packer
  //
  if (!theSkipRpc) {
  if (theConfig.getParameter<bool>("useRpcConnectionFile")) {
    theRpcPacker.init(es, edm::FileInPath(theConfig.getParameter<std::string>("rpcConnectionFile")).fullPath());
  } else {
    theRpcPacker.init(es); 
  } 
  }

  //
  // init CSC Link map
  //
  if (!theSkipCsc) theCscPacker.init();
}


void OmtfPacker::produce(edm::Event& ev, const edm::EventSetup& es)
{
  LogDebug ("OmtfPacker::produce") <<"Beginning To Pack Event: "<<++theEventCounter;

  //
  // prepare FedAmcRawsMap structure to collect Word64 raws from digi packers 
  //
  FedAmcRawsMap raws;
  std::list<unsigned int> amcIds = { 1, 3, 5, 7, 9, 11};
  std::list<unsigned int> fedIds = { 1380, 1381};
  for (auto & fedId : fedIds) { for (auto & amcId : amcIds) { raws[std::make_pair(fedId, amcId)]; } }

  //
  // DT raws
  //
  if (!theSkipDt) {
    edm::Handle<L1MuDTChambPhContainer> digiCollectionDTPh;
    ev.getByToken(theDtPhToken, digiCollectionDTPh);
    edm::Handle<L1MuDTChambThContainer> digiCollectionDTTh;
    ev.getByToken(theDtThToken, digiCollectionDTTh);
    theDtPacker.pack(digiCollectionDTPh.product(),digiCollectionDTTh.product(), raws);
  }

  //
  // csc raws
  //
  if (!theSkipCsc) {
    edm::Handle<CSCCorrelatedLCTDigiCollection> digiCollectionCSC;
    ev.getByToken(theCscToken,digiCollectionCSC);
    theCscPacker.pack(digiCollectionCSC.product(), raws);
  }

  //
  // rpc raws
  // 
  if (!theSkipRpc) {
    edm::Handle< RPCDigiCollection > digiCollectionRPC;
    ev.getByToken(theRpcToken,digiCollectionRPC);
    theRpcPacker.pack( digiCollectionRPC.product(), raws);
  }

  auto bxId  = ev.bunchCrossing();
  auto evtId = ev.id().event();
  auto orbit = ev.eventAuxiliary().orbitNumber();
  std::unique_ptr<FEDRawDataCollection> raw_coll(new FEDRawDataCollection());
  for (auto & fedId : fedIds) {

    //
    // assign formatted raws to feds
    //
    amc13::Packet amc13;
    for (const auto & it : raws) {
      if (fedId != it.first.first) continue;
      const std::vector<Word64> & amcData = it.second;
      unsigned int amcId = it.first.second;
      for (const auto & raw : amcData) {
        std::ostringstream dataStr;
        if (DataWord64::csc  == DataWord64::type(raw)) dataStr <<  CscDataWord64(raw);
        if (DataWord64::dt   == DataWord64::type(raw)) dataStr <<   DtDataWord64(raw);
        if (DataWord64::rpc  == DataWord64::type(raw)) dataStr <<  RpcDataWord64(raw);
        if (DataWord64::omtf == DataWord64::type(raw)) dataStr << MuonDataWord64(raw);
        LogTrace("")<<" fed: "<< fedId <<" amcId: "<<amcId<<" RAW DATA: " << dataStr.str() << std::endl;
      }
      amc13.add(amcId, 43981, evtId, orbit, bxId, amcData);
    }

    FEDRawData& fed_data = raw_coll->FEDData(fedId);

    const unsigned int slinkHeaderSize  = 8;
    const unsigned int slinkTrailerSize = 8;
    unsigned int size = (amc13.size()) * sizeof(Word64)+slinkHeaderSize+slinkTrailerSize;
    fed_data.resize(size);
    unsigned char * payload = fed_data.data();
    unsigned char * payload_start = payload;

    FEDHeader header(payload);
    const unsigned int evtType = 1;
    header.set(payload, evtType, evtId, bxId, fedId);

    amc13.write(ev, payload, slinkHeaderSize, size - slinkHeaderSize - slinkTrailerSize);

    payload += slinkHeaderSize;
    payload += amc13.size() * 8;

    FEDTrailer trailer(payload);
    trailer.set(payload, size / 8, evf::compute_crc(payload_start, size), 0, 0);
  }
  ev.put(std::move(raw_coll), theOutputTag);


}

}
using namespace omtf;
DEFINE_FWK_MODULE(OmtfPacker);


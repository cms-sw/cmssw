#include <iostream>
#include <iomanip>
#include <bitset>
#include <string>
#include <map>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
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

#include "EventFilter/L1TRawToDigi/interface/OmtfEleIndex.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfLinkMappingRpc.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfLinkMappingCsc.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"
#include "EventFilter/RPCRawToDigi/interface/RPCPackingModule.h"
#include "EventFilter/RPCRawToDigi/interface/DebugDigisPrintout.h"
#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"

#include "EventFilter/L1TRawToDigi/interface/OmtfRpcPacker.h"



using namespace Omtf;
namespace Omtf {

class OmtfPacker: public edm::stream::EDProducer<> {
public:

    ///Constructor
    OmtfPacker(const edm::ParameterSet& pset);

    ~OmtfPacker() {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void produce(edm::Event & ev, const edm::EventSetup& es) override;

    void beginRun(const edm::Run &run, const edm::EventSetup& es) override;

private:

  edm::ParameterSet theConfig;
  edm::InputTag dataLabel_;
  unsigned long theEventCounter;

  edm::EDGetTokenT<RPCDigiCollection> rpcToken_;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> cscToken_;
  edm::EDGetTokenT<L1MuDTChambThContainer> dtThToken_;
  edm::EDGetTokenT<L1MuDTChambPhContainer> dtPhToken_;

  MapCscDet2EleIndex       theCsc2Omtf;
  MapLBIndex2EleIndex      thePact2Omtf;
  const RPCReadOutMapping* thePactCabling;

  RpcPacker theRpcPacker;

};

OmtfPacker::OmtfPacker(const edm::ParameterSet& pset) : theConfig(pset), theEventCounter(0) {

  produces<FEDRawDataCollection>("OmtfPacker");

  rpcToken_ = consumes<RPCDigiCollection>(pset.getParameter<edm::InputTag>("rpcInputLabel"));
  cscToken_ = consumes<CSCCorrelatedLCTDigiCollection>(pset.getParameter<edm::InputTag>("cscInputLabel"));
  dtPhToken_ = consumes<L1MuDTChambPhContainer>(pset.getParameter<edm::InputTag>("dtPhInputLabel"));
  dtThToken_ = consumes<L1MuDTChambThContainer>(pset.getParameter<edm::InputTag>("dtThInputLabel"));
}

void OmtfPacker::beginRun(const edm::Run &run, const edm::EventSetup& es) {

  //
  // initialise RPC packer
  //
  if (theConfig.getParameter<bool>("useRpcConnectionFile")) {
    theRpcPacker.init(es, edm::FileInPath(theConfig.getParameter<std::string>("rpcConnectionFile")).fullPath());
  } else {
    theRpcPacker.init(es); 
  } 

/*
  //
  // initialise PACT cabling
  //
  edm::ESTransientHandle<RPCEMap> readoutMapping;
  es.get<RPCEMapRcd>().get(readoutMapping);
  thePactCabling = readoutMapping->convert();
  LogDebug("OmtfPacker") <<" Has PACT readout map, VERSION: " << thePactCabling->version() << std::endl;

  //
  // PACT <--> OMTF translation
  //
  RpcLinkMap omtfLink2Ele;
  if (theConfig.getParameter<bool>("useRpcConnectionFile")) {
    omtfLink2Ele.init( edm::FileInPath(theConfig.getParameter<std::string>("rpcConnectionFile")).fullPath());
  } else {
    omtfLink2Ele.init(es); 
  } 
  thePact2Omtf = translatePact2Omtf(omtfLink2Ele,thePactCabling);
*/

  //
  // init CSC Link map
  //
  theCsc2Omtf = mapCscDet2EleIndex();
}


void OmtfPacker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("rpcInputLabel",edm::InputTag(""));
  desc.add<edm::InputTag>("cscInputLabel",edm::InputTag(""));
  desc.add<edm::InputTag>("dtPhInputLabel",edm::InputTag(""));
  desc.add<edm::InputTag>("dtThInputLabel",edm::InputTag(""));
  desc.add<bool>("useRpcConnectionFile",false);
  desc.add<std::string>("rpcConnectionFile","");
  descriptions.add("omtfPacker",desc);
}

void OmtfPacker::produce(edm::Event& ev, const edm::EventSetup& es)
{
  bool debug = edm::MessageDrop::instance()->debugEnabled;
  theEventCounter++;
  if (debug) LogDebug ("OmtfPacker::produce") <<"Beginning To Pack Event: "<<theEventCounter;

  std::map< std::pair<unsigned int, unsigned int>, std::vector<Word64> > raws;
  std::list<unsigned int> amcIds = { 1, 3, 5, 7, 9, 11};
  std::list<unsigned int> fedIds = { 1380, 1381};
  for (auto & fedId : fedIds) { for (auto & amdId : amcIds) { raws[std::make_pair(fedId, amdId)]; } }

  //
  // DT raws
  //
  edm::Handle<L1MuDTChambPhContainer> digiCollectionDTPh;
  ev.getByToken(dtPhToken_, digiCollectionDTPh);
  const L1MuDTChambPhContainer& dtphDigisBMTF = *digiCollectionDTPh.product();

  edm::Handle<L1MuDTChambThContainer> digiCollectionDTTh;
  ev.getByToken(dtThToken_, digiCollectionDTTh);
  const L1MuDTChambThContainer& dtthDigisBMTF = *digiCollectionDTTh.product();

  for (const auto &  chDigi : *dtphDigisBMTF.getContainer() ) {
    if (abs(chDigi.whNum()) != 2) continue;
    if (chDigi.stNum() ==4) continue;
    DtDataWord64 data;
    data.st_phi_ = chDigi.phi();
    data.st_phib_ = chDigi.phiB();
    data.st_q_    = chDigi.code();
    int bxNumber = chDigi.bxNum();
    data.bxNum_ = (3+bxNumber);
    data.st_ = chDigi.stNum()-1;
    data.valid_ = 1;
    int bxCnt = (chDigi.BxCnt() >= 0 && chDigi.BxCnt() <=3) ? chDigi.BxCnt() : 0;
    data.bcnt_st_ = bxCnt;
    data.bcnt_e0_ = bxCnt;
    data.bcnt_e1_ = bxCnt;
    data.fiber_   = chDigi.Ts2Tag();
    unsigned int amc;
    unsigned int amc2=0;
    unsigned int fed = (chDigi.whNum()==-2)? 1380: 1381;
    if (chDigi.scNum()%2 !=0) {
      amc = chDigi.scNum();
      data.sector_ = 1;
    } else {
      amc = chDigi.scNum()+1;
      data.sector_ = 0;
      amc2 = (chDigi.scNum()+11)%12; // in this case data.sector_ should be 2, fixed later
    }
//    LogTrace("")<<" fed: "<< fed <<" amc: "<<amc<<" DT PH DATA: " << data << std::endl;
    raws[std::make_pair(fed,amc)].push_back(data.rawData);
    if (amc2 != 0) {
      data.sector_ = 2;
//      LogTrace("")<<" fed: "<< fed <<" amc: "<<amc2<<" DT PH DATA: " << data << std::endl;
      raws[std::make_pair(fed,amc2)].push_back(data.rawData);
    }
  }


  for (const auto &  chDigi : *dtthDigisBMTF.getContainer() ) {
    if (abs(chDigi.whNum()) != 2) continue;
    if (chDigi.stNum() ==4) continue;
    DtDataWord64 data;
    int bxNumber = chDigi.bxNum();
    data.bxNum_ = (3+bxNumber);
    data.st_ = chDigi.stNum()-1;
    data.valid_ = 1;
    unsigned int amc;
    unsigned int amc2=0;
    unsigned int fed = (chDigi.whNum()==-2)? 1380: 1381;
    if (chDigi.scNum()%2 !=0) {
      amc = chDigi.scNum();
      data.sector_ = 1;
    } else {
      amc = chDigi.scNum()+1;
      data.sector_ = 0;
      amc2 = (chDigi.scNum()+11)%12; // in this case data.sector_ should be 2, fixed later
    }
    unsigned int eta = 0;
    unsigned int etaQ = 0;
    for (unsigned int ipos=0; ipos <7; ipos++) {
      if (chDigi.position(ipos) >1 ) std::cout <<"DT TH position to ETA,  PROBLEM !!!!" << std::endl;
      if (chDigi.position(ipos)==1) eta |= (1 <<ipos);
      if (chDigi.quality(ipos)==1) etaQ |= (1 <<ipos);
    }
    data.eta_qbit_ = etaQ;
    data.eta_hit_  = eta;
    bool foundDigi = false;
    for (auto & raw : raws) {
      if (raw.first.first != fed) continue;
      if (raw.first.second != amc &&  raw.first.second != amc2) continue;
      auto & words = raw.second;
      for (auto & word : words) {
        if (DataWord64::dt != DataWord64::type(word)) continue;
        DtDataWord64 dataRaw(word);
        if (dataRaw.bxNum_ != data.bxNum_) continue;
        if (dataRaw.st_    != data.st_) continue;
        foundDigi = true;
        dataRaw.eta_qbit_ =  data.eta_qbit_;
        dataRaw.eta_hit_ =  data.eta_hit_;
        word = dataRaw.rawData;
//        LogTrace("")<<" fed: "<< fed <<" amc: "<<amc<<" DT TH DATA: " << dataRaw << std::endl;
      }
    }
    if (!foundDigi) {
//      LogTrace("")<<" fed: "<< fed <<" amc: "<<amc<<" DT TH DATA: " << data<< std::endl;
      raws[std::make_pair(fed,amc)].push_back(data.rawData);
      if (amc2 != 0) {
        data.sector_ = 2;
//        LogTrace("")<<" fed: "<< fed <<" amc: "<<amc2<<" DT TH DATA: " << data<< std::endl;
        raws[std::make_pair(fed,amc2)].push_back(data.rawData);
      }
    }
  }

  //
  // csc raws
  //
  edm::Handle<CSCCorrelatedLCTDigiCollection> digiCollectionCSC;
  ev.getByToken(cscToken_,digiCollectionCSC);
  const CSCCorrelatedLCTDigiCollection & cscDigis = *digiCollectionCSC.product();
  for (const auto & chDigis : cscDigis) {
    CSCDetId chamberId = CSCDetId(chDigis.first).chamberId();
    for (auto digi = chDigis.second.first; digi != chDigis.second.second; digi++) {
      CscDataWord64 data;
      data.hitNum_ = digi->getTrknmb();
      data.vp_ = digi->isValid();
      data.bxNum_ = digi->getBX()-3;
      data.halfStrip_ = digi->getStrip();
      data.clctPattern_ = digi->getPattern();
      data.keyWG_ = digi->getKeyWG();
      data.lr_ = digi->getBend();
      data.quality_ = digi->getQuality();
      auto im = theCsc2Omtf.find(chamberId);
      if (im != theCsc2Omtf.end()) {
        std::vector<EleIndex> links = {im->second.first, im->second.second};
        for (const auto & link : links) {
          unsigned int fed = link.fed();
          if (fed == 0) continue; 
          data.station_ = chamberId.station()-1;
          data.linkNum_ = link.link();
          data.cscID_ = chamberId.chamber()-(link.amc()-1)*6;
          unsigned int amc =  link.amc()*2-1;
          raws[std::make_pair(fed,amc)].push_back(data.rawData);
          LogTrace("") <<"ADDED RAW: fed: "<< fed <<" amc: "<<amc <<" CSC DATA: " << data<< std::endl;
        }
      }
    }
  }

  //
  // rpc raws
  // 
  edm::Handle< RPCDigiCollection > digiCollectionRPC;
  ev.getByToken(rpcToken_,digiCollectionRPC);
  theRpcPacker.pack( digiCollectionRPC.product(), raws);
/*
  edm::Handle< RPCDigiCollection > digiCollectionRPC;
  ev.getByToken(rpcToken_,digiCollectionRPC);
  LogTrace("")<<" HERE HERE !!! RPC PACKER" << rpcrawtodigi::DebugDigisPrintout()(digiCollectionRPC.product());
  for (int dcc=790; dcc <= 792; dcc++) {
    RPCRecordFormatter formatter(dcc, thePactCabling);
    std::vector<rpcrawtodigi::EventRecords> merged = RPCPackingModule::eventRecords(dcc,200, digiCollectionRPC.product(),formatter);
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
*/


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
//        DtDataWord64 data(raw);
          LogTrace("")<<" fed: "<< fedId <<" amcId: "<<amcId<<" RAW DATA: " << dataStr.str() << std::endl;
        }
      amc13.add(amcId, 43981, evtId, orbit, bxId, amcData);
    }

    FEDRawData& fed_data = raw_coll->FEDData(fedId);

    const unsigned int slinkHeaderSize_  = 8;
    const unsigned int slinkTrailerSize_ = 8;
    unsigned int size = (amc13.size()) * sizeof(Word64)+slinkHeaderSize_+slinkTrailerSize_;
    fed_data.resize(size);
    unsigned char * payload = fed_data.data();
    unsigned char * payload_start = payload;

    FEDHeader header(payload);
    const unsigned int evtType_ = 1;
    header.set(payload, evtType_, evtId, bxId, fedId);

    amc13.write(ev, payload, slinkHeaderSize_, size - slinkHeaderSize_ - slinkTrailerSize_);

    payload += slinkHeaderSize_;
    payload += amc13.size() * 8;

    FEDTrailer trailer(payload);
    trailer.set(payload, size / 8, evf::compute_crc(payload_start, size), 0, 0);
  }
  ev.put(std::move(raw_coll),"OmtfPacker");


}

};
DEFINE_FWK_MODULE(OmtfPacker);


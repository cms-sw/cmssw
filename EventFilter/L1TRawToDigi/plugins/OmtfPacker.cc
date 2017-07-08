#include <iostream>
#include <iomanip>
#include <bitset>
#include <string>
#include <map>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
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

#include "EventFilter/L1TRawToDigi/interface/OmtfRpcLinkMap.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"


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
  unsigned long eventCounter_;

  edm::EDGetTokenT<RPCDigiCollection> rpcToken_;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> cscToken_;
  edm::EDGetTokenT<L1MuDTChambThContainer> dtThToken_;
  edm::EDGetTokenT<L1MuDTChambPhContainer> dtPhToken_;

  std::map<EleIndex, LinkBoardElectronicIndex> omtf2rpc_;
  std::map<EleIndex, CSCDetId> omtf2csc_;

  const RPCReadOutMapping* theCabling;


};

OmtfPacker::OmtfPacker(const edm::ParameterSet& pset) : theConfig(pset), eventCounter_(0) {

  produces<FEDRawDataCollection>("OmtfPacker");

  rpcToken_ = consumes<RPCDigiCollection>(pset.getParameter<edm::InputTag>("rpcInputLabel"));
  cscToken_ = consumes<CSCCorrelatedLCTDigiCollection>(pset.getParameter<edm::InputTag>("cscInputLabel"));
  dtPhToken_ = consumes<L1MuDTChambPhContainer>(pset.getParameter<edm::InputTag>("dtPhInputLabel"));
  dtThToken_ = consumes<L1MuDTChambThContainer>(pset.getParameter<edm::InputTag>("dtThInputLabel"));
}

void OmtfPacker::beginRun(const edm::Run &run, const edm::EventSetup& es) {
  edm::ESTransientHandle<RPCEMap> readoutMapping;
  es.get<RPCEMapRcd>().get(readoutMapping);
  const RPCReadOutMapping * cabling= readoutMapping->convert();
  theCabling = cabling;
  std::cout << "HERE !!!! " << std::endl;
  LogDebug("OmtfPacker") <<" Has readout map, VERSION: " << cabling->version() << std::endl;

  RpcLinkMap omtfLink2Ele;

//  if (theConfig.getParameter<bool>("useRpcConnectionFile")) {
//    edm::FileInPath fip(theConfig.getParameter<string>("rpcConnectionFile"));
//    omtfLink2Ele.init(fip.fullPath());
//  } else {
    edm::ESHandle<RPCAMCLinkMap> amcMapping;
    es.get<RPCOMTFLinkMapRcd>().get(amcMapping);
    omtfLink2Ele.init(amcMapping->getMap());
//  }

  std::vector<const DccSpec*> dccs = cabling->dccList();
  for (std::vector<const DccSpec*>::const_iterator it1= dccs.begin(); it1!= dccs.end(); ++it1) {
    const std::vector<TriggerBoardSpec> & rmbs = (*it1)->triggerBoards();
    for (std::vector<TriggerBoardSpec>::const_iterator it2 = rmbs.begin(); it2 != rmbs.end(); ++it2) {
      const  std::vector<LinkConnSpec> & links = it2->linkConns();
      for (std::vector<LinkConnSpec>::const_iterator it3 = links.begin(); it3 != links.end(); ++it3) {
        const  std::vector<LinkBoardSpec> & lbs = it3->linkBoards();
        for (std::vector<LinkBoardSpec>::const_iterator it4=lbs.begin(); it4 != lbs.end(); ++it4) {

          try {
            std::string lbNameCH = it4->linkBoardName();
            std::string lbName = lbNameCH.substr(0,lbNameCH.size()-4);
            const std::vector<EleIndex> & omtfEles = omtfLink2Ele.omtfEleIndex(lbName);
//          std::cout <<"  isOK ! " <<  it4->linkBoardName() <<" has: " << omtfEles.size() << " first: "<< omtfEles[0] << std::endl;
            LinkBoardElectronicIndex rpcEle = { (*it1)->id(), it2->dccInputChannelNum(), it3->triggerBoardInputNumber(), it4->linkBoardNumInLink()};
            for ( const auto & omtfEle : omtfEles ) omtf2rpc_[omtfEle]= rpcEle;
          }
          catch(...) { ; } // std::cout << "exception! "<<it4->linkBoardName()<< std::endl; }
        }
      }
    }
  }
  LogTrace(" ") << " SIZE OF OMTF to RPC map  is: " << omtf2rpc_.size() << std::endl;

  //
  // init CSC Link map
  //
  omtf2csc_.clear();
  for (unsigned int fed=1380; fed<=1381; fed++) {
    //Endcap label. 1=forward (+Z); 2=backward (-Z)
    unsigned int endcap = (fed==1380) ? 2 : 1;
    for (unsigned int amc=1;    amc<=6; amc++) {
      for (unsigned int link=0; link <=34; link++) {
        unsigned int stat=0;
        unsigned int ring=0;
        unsigned int cham=0;
        switch (link) {
          case ( 0) : { stat=1; ring=2; cham=3; break;} //  (0,  9, 2, 3 ), --channel_0  OV1A_4 chamber_ME1/2/3  layer_9 input 2, 3
          case ( 1) : { stat=1; ring=2; cham=4; break;} //  (1,  9, 4, 5 ), --channel_1  OV1A_5 chamber_ME1/2/4  layer_9 input 4, 5
          case ( 2) : { stat=1; ring=2; cham=5; break;} //  (2,  9, 6, 7 ), --channel_2  OV1A_6 chamber_ME1/2/5  layer_9 input 6, 7
          case ( 3) : { stat=1; ring=3; cham=3; break;} //  (3,  6, 2, 3 ), --channel_3  OV1A_7 chamber_ME1/3/3  layer_6 input 2, 3
          case ( 4) : { stat=1; ring=3; cham=4; break;} //  (4,  6, 4, 5 ), --channel_4  OV1A_8 chamber_ME1/3/4  layer_6 input 4, 5
          case ( 5) : { stat=1; ring=3; cham=5; break;} //  (5,  6, 6, 7 ), --channel_5  OV1A_9 chamber_ME1/3/5  layer_6 input 6, 7
          case ( 6) : { stat=1; ring=2; cham=6; break;} //  (6,  9, 8, 9 ), --channel_6  OV1B_4 chamber_ME1/2/6  layer_9 input 8, 9
          case ( 7) : { stat=1; ring=2; cham=7; break;} //  (7,  9, 10,11), --channel_7  OV1B_5 chamber_ME1/2/7  layer_9 input 10,11
          case ( 8) : { stat=1; ring=2; cham=8; break;} //  (8,  9, 12,13), --channel_8  OV1B_6 chamber_ME1/2/8  layer_9 input 12,13
          case ( 9) : { stat=1; ring=3; cham=6; break;} //  (9,  6, 8, 9 ), --channel_9  OV1B_7 chamber_ME1/3/6  layer_6 input 8, 9
          case (10) : { stat=1; ring=3; cham=7; break;} //  (10, 6, 10,11), --channel_10 OV1B_8 chamber_ME1/3/7  layer_6 input 10,11
          case (11) : { stat=1; ring=3; cham=8; break;} //  (11, 6, 12,13), --channel_11 OV1B_9 chamber_ME1/3/8  layer_6 input 12,13
          case (12) : { stat=2; ring=2; cham=3; break;} //  (12, 7, 2, 3 ), --channel_0  OV2_4  chamber_ME2/2/3  layer_7 input 2, 3
          case (13) : { stat=2; ring=2; cham=4; break;} //  (13, 7, 4, 5 ), --channel_1  OV2_5  chamber_ME2/2/4  layer_7 input 4, 5
          case (14) : { stat=2; ring=2; cham=5; break;} //  (14, 7, 6, 7 ), --channel_2  OV2_6  chamber_ME2/2/5  layer_7 input 6, 7
          case (15) : { stat=2; ring=2; cham=6; break;} //  (15, 7, 8, 9 ), --channel_3  OV2_7  chamber_ME2/2/6  layer_7 input 8, 9
          case (16) : { stat=2; ring=2; cham=7; break;} //  (16, 7, 10,11), --channel_4  OV2_8  chamber_ME2/2/7  layer_7 input 10,11
          case (17) : { stat=2; ring=2; cham=8; break;} //  (17, 7, 12,13), --channel_5  OV2_9  chamber_ME2/2/8  layer_7 input 12,13
          case (18) : { stat=3; ring=2; cham=3; break;} //  (18, 8, 2, 3 ), --channel_6  OV3_4  chamber_ME3/2/3  layer_8 input 2, 3
          case (19) : { stat=3; ring=2; cham=4; break;} //  (19, 8, 4, 5 ), --channel_7  OV3_5  chamber_ME3/2/4  layer_8 input 4, 5
          case (20) : { stat=3; ring=2; cham=5; break;} //  (20, 8, 6, 7 ), --channel_8  OV3_6  chamber_ME3/2/5  layer_8 input 6, 7
          case (21) : { stat=3; ring=2; cham=6; break;} //  (21, 8, 8, 9 ), --channel_9  OV3_7  chamber_ME3/2/6  layer_8 input 8, 9
          case (22) : { stat=3; ring=2; cham=7; break;} //  (22, 8, 10,11), --channel_10 OV3_8  chamber_ME3/2/7  layer_8 input 10,11
          case (23) : { stat=3; ring=2; cham=8; break;} //  (23, 8, 12,13), --channel_11 OV3_9  chamber_ME3/2/8  layer_8 input 12,13
          case (24) : { stat=4; ring=2; cham=3; break;} //--(24,  ,      ), --channel_3  OV4_4  chamber_ME4/2/3  layer   input
          case (25) : { stat=4; ring=2; cham=4; break;} //--(25,  ,      ), --channel_4  OV4_5  chamber_ME4/2/4  layer   input
          case (26) : { stat=4; ring=2; cham=5; break;} //--(26,  ,      ), --channel_5  OV4_6  chamber_ME4/2/5  layer   input
          case (27) : { stat=4; ring=2; cham=6; break;} //--(27,  ,      ), --channel_7  OV4_7  chamber_ME4/2/6  layer   input
          case (28) : { stat=4; ring=2; cham=7; break;} //--(28,  ,      ), --channel_8  OV4_8  chamber_ME4/2/7  layer   input
          case (29) : { stat=4; ring=2; cham=8; break;} //--(29,  ,      ), --channel_9  OV4_9  chamber_ME4/2/8  layer   input
          case (30) : { stat=1; ring=2; cham=2; break;} //  (30, 9, 0, 1 ), --channel_0  OV1B_6 chamber_ME1/2/2  layer_9 input 0, 1
          case (31) : { stat=1; ring=3; cham=2; break;} //  (31, 6, 0, 1 ), --channel_1  OV1B_9 chamber_ME1/3/2  layer_6 input 0, 1
          case (32) : { stat=2; ring=2; cham=2; break;} //  (32, 7, 0, 1 ), --channel_2  OV2_9  chamber_ME2/2/2  layer_7 input 0, 1
          case (33) : { stat=3; ring=2; cham=2; break;} //  (33, 8, 0, 1 ), --channel_3  ON3_9  chamber_ME3/2/2  layer_8 input 0, 1
          case (34) : { stat=4; ring=2; cham=2; break;} //--(34,  ,      ), --channel_4  ON4_9  chamber_ME4/2/2  layer   input
          default   : { stat=0; ring=0; cham=0; break;}
        }
        if (ring !=0) {
          int chamber = cham+(amc-1)*6;
          if (chamber > 36) chamber -= 36;
          CSCDetId cscDetId(endcap, stat, ring, chamber);
//          std::cout <<" INIT CSC DET ID: "<< cscDetId << std::endl;
         EleIndex omtfEle(fed, amc, link);
          omtf2csc_[omtfEle]=cscDetId;
        }
      }
    }
  }
}


void OmtfPacker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("rpcInputLabel",edm::InputTag(""));
  desc.add<edm::InputTag>("cscInputLabel",edm::InputTag(""));
  desc.add<edm::InputTag>("dtPhInputLabel",edm::InputTag(""));
  desc.add<edm::InputTag>("dtThInputLabel",edm::InputTag(""));
  desc.add<bool>("useRpcConnectionFile",bool(false));
  desc.add<std::string>("rpcConnectionFile",std::string("EventFilter/L1TRawToDigi/data/OmtfRpcLinksMap.txt"));
  descriptions.add("omtfPacker",desc);
}

void OmtfPacker::produce(edm::Event& ev, const edm::EventSetup& es)
{
  bool debug = edm::MessageDrop::instance()->debugEnabled;
  eventCounter_++;
  if (debug) LogDebug ("OmtfPacker::produce") <<"Beginning To Pack Event: "<<eventCounter_;

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
    auto rawId = chDigis.first;
    CSCDetId cscDetId(rawId);
    unsigned int fed = (cscDetId.zendcap()==-1)? 1380: 1381;
 //   std::cout <<"--------------"<< std::endl;
 //   std::cout <<"CSC DET ID: "<< cscDetId << std::endl;
    unsigned int amc = 0;
    for (auto digi = chDigis.second.first; digi != chDigis.second.second; digi++) {
 //     std::cout << *digi << std::endl;
      CscDataWord64 data;
      data.hitNum_ = digi->getTrknmb();
      data.vp_ = digi->isValid();
      data.bxNum_ = digi->getBX()-3;
      data.halfStrip_ = digi->getStrip();
      data.clctPattern_ = digi->getPattern();
      data.keyWG_ = digi->getKeyWG();
      data.lr_ = digi->getBend();
      data.quality_ = digi->getQuality();
      for (const auto & im : omtf2csc_) {
        if (im.second == rawId) {
  //        LogTrace("")<<" FOUND ELE INDEX " << im.first;
          data.station_ = cscDetId.station()-1;
          data.linkNum_ = im.first.link();
          data.cscID_ = cscDetId.chamber()-(im.first.amc()-1)*6;
          amc =  im.first.amc()*2-1;
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
  LogTrace("")<<" HERE HERE !!! RPC PACKER" << rpcrawtodigi::DebugDigisPrintout()(digiCollectionRPC.product());
  for (int dcc=790; dcc <= 792; dcc++) {
    RPCRecordFormatter formatter(dcc, theCabling);
    std::vector<rpcrawtodigi::EventRecords> merged = RPCPackingModule::eventRecords(dcc,200, digiCollectionRPC.product(),formatter);
    LogTrace("") << " SIZE OF MERGED, for DCC="<<dcc<<" is: "<<merged.size()<<std::endl;
    for (const auto & rpcEvent : merged) {
      RpcDataWord64 data;
//
      data.bxNum_ =  rpcEvent.dataToTriggerDelay();
      data.frame1_ = rpcEvent.recordCD().data();
      LinkBoardElectronicIndex rpcEle = { dcc, rpcEvent.recordSLD().rmb(),  rpcEvent.recordSLD().tbLinkInputNumber(), rpcEvent.recordCD().lbInLink() };
      for ( const auto & omtf2rpc : omtf2rpc_) {
        if (   omtf2rpc.second.dccId                == rpcEle.dccId
            && omtf2rpc.second.dccInputChannelNum == rpcEle.dccInputChannelNum
            && omtf2rpc.second.tbLinkInputNum     == rpcEle.tbLinkInputNum) {
          data.linkNum_ = omtf2rpc.first.link();
          std::cout << "KUKUK " << std::endl;
          raws[std::make_pair(omtf2rpc.first.fed(), omtf2rpc.first.amc()*2-1)].push_back(data.rawData);
        }
      }
      if (data.linkNum_==0) LogTrace("")<< " could not find omtfIndex for rpcEle : "<<rpcEle.print();
    }

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


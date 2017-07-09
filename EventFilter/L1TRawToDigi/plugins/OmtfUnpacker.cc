// system include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <bitset>
#include <map>
#include <string>

// user include files
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/CRC16.h"


#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "EventFilter/L1TRawToDigi/interface/AMC13Spec.h"
#include "EventFilter/L1TRawToDigi/interface/AMCSpec.h"

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

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

#include "EventFilter/L1TRawToDigi/interface/OmtfDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfCscDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfDtDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfRpcDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfMuonDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfEleIndex.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfLinkMappingRpc.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfLinkMappingCsc.h"

#include "EventFilter/L1TRawToDigi/interface/OmtfRpcUnpacker.h"

using namespace Omtf;
namespace Omtf {

class OmtfUnpacker: public edm::stream::EDProducer<> {
public:

  ///Constructor
    OmtfUnpacker(const edm::ParameterSet& pset);

  ~OmtfUnpacker() {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event & ev, const edm::EventSetup& es) override;

  void beginRun(const edm::Run &run, const edm::EventSetup& es) override;

private:

  edm::InputTag dataLabel_;
  unsigned long eventCounter_;

  edm::EDGetTokenT<FEDRawDataCollection> fedToken_;

  std::map<EleIndex, LinkBoardElectronicIndex> omtf2rpc_;
  std::map<EleIndex, CSCDetId> omtf2csc_;
  
  const RPCReadOutMapping* theCabling;
  RpcUnpacker theRpcUnpacker;

  edm::ParameterSet theConfig;
  std::string theOutputTag;
};

OmtfUnpacker::OmtfUnpacker(const edm::ParameterSet& pset) : theConfig(pset) {
  theOutputTag = pset.exists("outputTag") ? pset.getParameter<std::string>("outputTag") : "OmtfUnpacker" ; 
  produces<RPCDigiCollection>(theOutputTag);
  produces<CSCCorrelatedLCTDigiCollection>(theOutputTag);
  produces<l1t::RegionalMuonCandBxCollection >(theOutputTag);
  produces<L1MuDTChambPhContainer>(theOutputTag);
  produces<L1MuDTChambThContainer>(theOutputTag);

  fedToken_ = consumes<FEDRawDataCollection>(pset.getParameter<edm::InputTag>("InputLabel"));
}


void OmtfUnpacker::beginRun(const edm::Run &run, const edm::EventSetup& es) {
/*
  edm::ESTransientHandle<RPCEMap> readoutMapping;
  es.get<RPCEMapRcd>().get(readoutMapping);
  const RPCReadOutMapping * cabling= readoutMapping->convert();
  theCabling = cabling;
  LogDebug("OmtfUnpacker") <<" Has readout map, VERSION: " << cabling->version() << std::endl;


  RpcLinkMap omtfLink2Ele;
  if (theConfig.getParameter<bool>("useRpcConnectionFile")) {
    omtfLink2Ele.init( edm::FileInPath(theConfig.getParameter<std::string>("rpcConnectionFile")).fullPath());
  } else {
    omtfLink2Ele.init(es);
  }
  omtf2rpc_ = translateOmtf2Pact(omtfLink2Ele,theCabling);
*/

  if (theConfig.getParameter<bool>("useRpcConnectionFile")) {
    theRpcUnpacker.init(es, edm::FileInPath(theConfig.getParameter<std::string>("rpcConnectionFile")).fullPath());
  } else {
    theRpcUnpacker.init(es);
  }



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
  LogTrace(" ") << " SIZE OF OMTF to CSC map  is: " << omtf2csc_.size() << std::endl;

}

void OmtfUnpacker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel",edm::InputTag("rawDataCollector"));
  desc.add<bool>("useRpcConnectionFile",false);
  desc.add<std::string>("rpcConnectionFile","");
  desc.add<std::string>("outputTag","");
  descriptions.add("omtfUnpacker",desc);
}

void OmtfUnpacker::produce(edm::Event& event, const edm::EventSetup& setup)
{
  bool debug = edm::MessageDrop::instance()->debugEnabled;
  eventCounter_++;
  if (debug) LogDebug ("OmtfUnpacker::produce") <<"Beginning To Unpack Event: "<<eventCounter_;

  edm::Handle<FEDRawDataCollection> allFEDRawData;
  event.getByToken(fedToken_,allFEDRawData);

  auto producedRPCDigis = std::make_unique<RPCDigiCollection>();
  auto producedCscLctDigis = std::make_unique<CSCCorrelatedLCTDigiCollection>();
  auto producedMuonDigis = std::make_unique<l1t::RegionalMuonCandBxCollection>(); 
  producedMuonDigis->setBXRange(-3,3);
  auto producedDTPhDigis = std::make_unique<L1MuDTChambPhContainer>();
  auto producedDTThDigis = std::make_unique<L1MuDTChambThContainer>();
  std::vector<L1MuDTChambPhDigi> phi_Container;
  std::vector<L1MuDTChambThDigi> the_Container;

  for (int fedId= 1380; fedId<= 1381; ++fedId) {

    const FEDRawData & rawData = allFEDRawData->FEDData(fedId);
    int nWords = rawData.size()/sizeof(Word64);
    LogTrace("") <<"FED : " << fedId <<" words: " << nWords;
    if (nWords==0) continue;

    int triggerBX =0;
   
    //
    // FED header
    // Expect one FED header + AMC13 headers.
    const Word64* header = reinterpret_cast<const Word64* >(rawData.data());
    FEDHeader fedHeader( reinterpret_cast<const unsigned char*>(header));
    if (!fedHeader.check()) { LogTrace("") <<" ** PROBLEM **, header.check() failed, break"; break; }
    if ( fedHeader.sourceID() != fedId) {
      LogTrace("") <<" ** PROBLEM **, fedHeader.sourceID() != fedId"
          << "fedId = " << fedId<<" sourceID="<<fedHeader.sourceID();
    }
    triggerBX = fedHeader.bxID();
    if (debug) {
      std::ostringstream str;
      str <<"  header: "<< *reinterpret_cast<const std::bitset<64>*> (header) << std::endl;
      str <<"  header triggerType: " << fedHeader.triggerType()<< std::endl;
      str <<"  header lvl1ID:      " << fedHeader.lvl1ID() << std::endl;
      str <<"  header bxID:        " << fedHeader.bxID() << std::endl;
      str <<"  header sourceID:    " << fedHeader.sourceID() << std::endl;
      str <<"  header version:     " << fedHeader.version() << std::endl;
      str <<"  header more :     " << fedHeader.moreHeaders() << std::endl;
      str << " triggerBx "<< triggerBX << std::endl;
      LogTrace("") << str.str();
    }
    // AMC13 headers


    // 
    // FED trailed
    //
    const Word64* trailer=reinterpret_cast<const Word64* >(rawData.data())+(nWords-1); trailer++;
    bool moreTrailers = true;
    while (moreTrailers) {
      trailer--;
      FEDTrailer fedTrailer(reinterpret_cast<const unsigned char*>(trailer));
      if ( !fedTrailer.check()) {
        if (debug) LogTrace("") <<" ** PROBLEM **, trailer.check() failed, break";
        break;
      }
      if ( fedTrailer.lenght()!= nWords) {
        if (debug) LogTrace("")<<" ** PROBLEM **, fedTrailer.lenght()!= nWords, break";
        break;
      }
      moreTrailers = fedTrailer.moreTrailers();
      if (debug) {
        std::ostringstream str;
        str <<" trailer: "<<  *reinterpret_cast<const std::bitset<64>*> (trailer) << std::endl;
        str <<"  trailer lenght:    "<<fedTrailer.lenght()<< std::endl;
        str <<"  trailer crc:       "<<fedTrailer.crc()<< std::endl;
        str <<"  trailer evtStatus: "<<fedTrailer.evtStatus()<< std::endl;
        str <<"  trailer ttsBits:   "<<fedTrailer.ttsBits()<< std::endl;
        LogTrace("") << str.str();
      }
    }

    if (debug) {
      std::ostringstream str;
      for (const Word64* word = header+1; word != trailer; word++) {
        str<<"    data: "<<*reinterpret_cast<const std::bitset<64>*>(word) << std::endl;
      }
      LogTrace("") << str.str();
    }

    //
    // AMC13 header
    //
    const Word64* headerAmc13raw = header+1;
    amc13::Header headerAmc13(headerAmc13raw);
    if (debug) {
      std::ostringstream str;
      str <<" headerAMC13:  "<<  *reinterpret_cast<const std::bitset<64>*> (headerAmc13raw) << std::endl;
      str <<" amc13 check:  "<< headerAmc13.check() << std::endl;
      str <<" amc13 format: "<< headerAmc13.getFormatVersion() << std::endl;
      str <<" amc13 nAMCs:  "<< headerAmc13.getNumberOfAMCs() << std::endl;
      str <<" amc13 orbit:  "<< headerAmc13.getOrbitNumber() << std::endl;
      LogTrace("") << str.str();
    }


    //unsigned int nAMCs = headerAmc13.getNumberOfAMCs();
    //for (unsigned int iAMC = 0; iAMC <nAMCs; iAMC++) {
    // const Word64* raw = header+1 +(iAMC+1);
    //}
   

    //
    // AMC13 trailer
    //
    const Word64* trailerAmc13raw = trailer-1;
    amc13::Trailer trailerAmc13(trailerAmc13raw);
    if (debug) {
      std::ostringstream str;
      str <<" trailerAMC13:  "<<  *reinterpret_cast<const std::bitset<64>*> (trailerAmc13raw) << std::endl;
      str <<" crc:  "<< trailerAmc13.getCRC() << std::endl;
      str <<" block: "<< trailerAmc13.getBlock() << std::endl;
      str <<" LV1ID:  "<< trailerAmc13.getLV1ID() << std::endl;
      str <<" BX:  "<< trailerAmc13.getBX() << std::endl;
      LogTrace("") << str.str();
    }

    amc13::Packet packetAmc13;
    if (!packetAmc13.parse( header, header+1, nWords-2, fedHeader.lvl1ID(), fedHeader.bxID(), 1, 0)) {
      edm::LogError("OMTF") << "Could not extract AMC13 Packet.";
      return;
    } 
//    std::cout <<"AMC13 Packet: "<< packetAmc13.blocks() << " size "<<packetAmc13.size() << std::endl;  
    unsigned int blockNum=0;
    for (auto amc: packetAmc13.payload()) {
//      amc.finalize(fedHeader.lvl1ID(), fedHeader.bxID(), true, false);
      amc::BlockHeader bh =  amc.blockHeader(); 
      if (debug) {
        std::ostringstream str;
        str <<" ----------- #"<<blockNum++ << std::endl;
        str <<" blockheader:  "<<  std::bitset<64>(bh.raw()) << std::endl;
        str <<" boardID:  "<< bh.getBoardID() << std::endl;
        str <<" amcNumber:  "<< bh.getAMCNumber() << std::endl;
        str <<" size:  "<< bh.getSize(); // << std::endl;
        LogTrace("") << str.str();
      }

      //
      // AMC header
      //
      amc::Header headerAmc = amc.header(); 
      if (debug) {
        std::ostringstream str;
        str <<" AMC header[0]:  "<<  std::bitset<64>(headerAmc.raw()[0]) << std::endl;
        str <<" AMC header[1]:  "<<  std::bitset<64>(headerAmc.raw()[1]) << std::endl;
        str <<" AMC number:     "<< headerAmc.getAMCNumber(); // << std::endl;
        LogTrace("") << str.str();
      }


      //
      // AMC trailer
      //
      //amc::Trailer trailerAmc = amc.trailer();              //this is the expected way but does not work 
      amc::Trailer trailerAmc(amc.data().get()+amc.size()-1); //FIXME: the above is prefered but this works (CMSSW900)
      if (debug) {
        std::ostringstream str;
        str <<" AMC trailer:  "<<  std::bitset<64>(trailerAmc.raw()) << std::endl;
        str <<" getLV1ID:     "<< trailerAmc.getLV1ID() << std::endl;
        str <<" size:         "<< trailerAmc.getSize()  << std::endl;
        LogTrace("") << str.str();
      }

      //
      // AMC payload
      //
      const auto & payload64 = amc.data();
      const Word64* word = payload64.get();
      for (unsigned int iWord= 1; iWord<= amc.size(); iWord++, word++) {
        if (iWord<=2 ) continue; // two header words for each AMC
        if (iWord==amc.size() ) continue; // trailer for each AMC 

        LogTrace("") <<" payload: " <<  *reinterpret_cast<const std::bitset<64>*>(word);
        DataWord64::Type recordType = DataWord64::type(*word); 


        //
        // RPC data
        // 
        if (DataWord64::rpc==recordType) {
          theRpcUnpacker.unpack(triggerBX, fedHeader.sourceID(), bh.getAMCNumber()/2+1, RpcDataWord64(*word), producedRPCDigis.get());
        }
/*
        if (DataWord64::rpc==recordType) {
          RpcDataWord64 data(*word);
          LogTrace("") << data;
  
         EleIndex omtfEle(fedHeader.sourceID(), bh.getAMCNumber()/2+1, data.linkNum());
          LinkBoardElectronicIndex rpcEle = omtf2rpc_.at(omtfEle);
          RPCRecordFormatter formater(rpcEle.dccId, theCabling);
  
  
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
  
            if (records.complete()) formater.recordUnpack( records,  producedRPCDigis.get(), 0,0);
          }
        }
*/



        //
        // CSC data
        //
        if (DataWord64::csc==recordType) {
          CscDataWord64   data(*word);
          EleIndex omtfEle(fedHeader.sourceID(), bh.getAMCNumber()/2+1, data.linkNum());
          std::map<EleIndex,CSCDetId>::const_iterator icsc = omtf2csc_.find(omtfEle);
          if (icsc==omtf2csc_.end()) {LogTrace(" ") <<" CANNOT FIND key: " << omtfEle << std::endl; continue; }
          CSCDetId cscId = omtf2csc_[omtfEle];
          LogTrace("") <<"OMTF->CSC "<<cscId << std::endl; 
          LogTrace("") << data << std::endl;
          if (data.linkNum() >=30) {LogTrace(" ")<<" data from overlap, skip digi "<< std::endl; continue;}
          CSCCorrelatedLCTDigi digi(data.hitNum(), //trknmb
                                    data.valid(), 
                                    data.quality(),
                                    data.wireGroup(),
                                    data.halfStrip(),
                                    data.clctPattern(),
                                    data.bend(),
                                    data.bxNum()+3) ;
          LogTrace("") << digi << std::endl;
          producedCscLctDigis->insertDigi( cscId, digi); 

        } 

        //
        // OMTF (muon) data
        //
        if (DataWord64::omtf==recordType) {
          MuonDataWord64   data(*word);
          LogTrace("") <<"OMTF->MUON " << std::endl;
          LogTrace("") << data << std::endl;
          l1t::tftype  overlap = (fedId==1380) ? l1t::tftype::omtf_neg :  l1t::tftype::omtf_pos;
          unsigned int iProcessor = bh.getAMCNumber()/2;   //0-5 
          l1t::RegionalMuonCand digi;
          digi.setHwPt(data.pT());
          digi.setHwEta(data.eta());
          digi.setHwPhi(data.phi());
          digi.setHwSign(data.ch());
          digi.setHwSignValid(data.vch());
          digi.setHwQual(data.quality());
          std::map<int, int> trackAddr;
          trackAddr[0]=data.layers();
          trackAddr[1]=0;
          trackAddr[2]=data.weight_lowBits();
          digi.setTrackAddress(trackAddr);
          digi.setTFIdentifiers(iProcessor, overlap);
          int bx = data.bxNum()-3;
          LogTrace("")  <<"OMTF Muon, BX="<<bx<<", hwPt="<<digi.hwPt()<< std::endl;
          if(std::abs(bx) <= 3) producedMuonDigis->push_back(bx,digi);
        }


        //
        // DT data
        //
        if (DataWord64::dt==recordType) {
          DtDataWord64   data(*word);
          LogTrace("") <<"HERE OMTF->DT " << std::endl;
          LogTrace("") << data << std::endl;
          if (data.sector()==0) {
             LogTrace("") << "...data skipped, since from oberlaping chambers."<< std::endl;
             continue; // skip signal from chamber fiber exchange
          }
          int bx = data.bxNum()-3;
          int whNum = (fedId==1380) ? -2 : 2;
          int sector =   (bh.getAMCNumber()/2)*2 + data.sector();
          if (sector==12) sector=0;
          int station =  data.station()+1;
          LogTrace("") <<"DT_AMC#  "<<bh.getAMCNumber()<<" RAW_SECTOR: "<<data.sector()<<" DT_SECTOR: "<<sector<<std::endl;
          phi_Container.push_back( L1MuDTChambPhDigi( bx, whNum, sector, station,
                                   data.phi(), data.phiB(), data.quality(), 
                                   data.fiber(),               // utag/Ts2Tag 
                                   data.bcnt_st())); //ucnt/BxCnt  
          int pos[7];
          int posQual[7];
          for (unsigned int i=0; i<7; i++) { pos[i] = (data.eta() >> i & 1); posQual[i] = (data.etaQuality() >> i & 1); }
          if (data.eta()) LogTrace("") <<"HERE DATA DT ETA";
          if(data.eta())the_Container.push_back(L1MuDTChambThDigi(bx,whNum, sector, station, pos, posQual)); 
        }

      }
    }         

  } 
  event.put(std::move(producedRPCDigis),theOutputTag);
  event.put(std::move(producedCscLctDigis),theOutputTag);
  event.put(std::move(producedMuonDigis),theOutputTag); 
  producedDTPhDigis->setContainer(phi_Container);
  event.put(std::move(producedDTPhDigis),theOutputTag);
  producedDTThDigis->setContainer(the_Container);
  event.put(std::move(producedDTThDigis),theOutputTag);

}

};

DEFINE_FWK_MODULE(OmtfUnpacker);

// system include files
#include <iostream>
#include <iomanip>
#include <sstream>
#include <bitset>
#include <string>

// user include files
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
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

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "DataFormats/L1TMuon/interface/OMTF/OmtfDataWord64.h"
#include "DataFormats/L1TMuon/interface/OMTF/OmtfCscDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfDtDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfRpcDataWord64.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfMuonDataWord64.h"

#include "EventFilter/L1TRawToDigi/interface/OmtfRpcUnpacker.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfCscUnpacker.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfDtUnpacker.h"
#include "EventFilter/L1TRawToDigi/interface/OmtfMuonUnpacker.h"

namespace omtf {

class OmtfUnpacker: public edm::stream::EDProducer<> {
public:

  OmtfUnpacker(const edm::ParameterSet& pset);

  ~OmtfUnpacker() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event & ev, const edm::EventSetup& es) override;

  void beginRun(const edm::Run &run, const edm::EventSetup& es) override;

private:

  unsigned long theEventCounter;

  edm::EDGetTokenT<FEDRawDataCollection> theFedDataToken;

  RpcUnpacker  theRpcUnpacker;
  CscUnpacker  theCscUnpacker;
  DtUnpacker   theDtUnpacker;
  MuonUnpacker theMuonUnpacker;

  edm::ParameterSet theConfig;
  std::string theOutputTag;

  bool theSkipRpc, theSkipCsc, theSkipDt, theSkipMuon;
};


OmtfUnpacker::OmtfUnpacker(const edm::ParameterSet& pset) : theConfig(pset) {
  theOutputTag = pset.getParameter<std::string>("outputTag");

  produces<RPCDigiCollection>(theOutputTag);
  produces<CSCCorrelatedLCTDigiCollection>(theOutputTag);
  produces<l1t::RegionalMuonCandBxCollection >(theOutputTag);
  produces<L1MuDTChambPhContainer>(theOutputTag);
  produces<L1MuDTChambThContainer>(theOutputTag);

  theSkipDt   = pset.getParameter<bool>("skipDt");
  theSkipRpc  = pset.getParameter<bool>("skipRpc");
  theSkipCsc  = pset.getParameter<bool>("skipCsc");
  theSkipMuon = pset.getParameter<bool>("skipMuon");

  theFedDataToken = consumes<FEDRawDataCollection>(pset.getParameter<edm::InputTag>("inputLabel"));
}

void OmtfUnpacker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputLabel",edm::InputTag("rawDataCollector"));
  desc.add<bool>("skipRpc",false);
  desc.add<bool>("skipCsc",false);
  desc.add<bool>("skipDt",false);
  desc.add<bool>("skipMuon",false);
  desc.add<bool>("useRpcConnectionFile",false);
  desc.add<std::string>("rpcConnectionFile","");
  desc.add<std::string>("outputTag","");
  descriptions.add("omtfUnpacker",desc);
}


void OmtfUnpacker::beginRun(const edm::Run &run, const edm::EventSetup& es) {

  //
  // rpc unpacker 
  //
  if (!theSkipRpc) {
  if (theConfig.getParameter<bool>("useRpcConnectionFile")) {
    theRpcUnpacker.init(es, edm::FileInPath(theConfig.getParameter<std::string>("rpcConnectionFile")).fullPath());
  } else {
    theRpcUnpacker.init(es);
  }
  }


  //
  // csc unpacker
  //
  if (!theSkipCsc) theCscUnpacker.init();

}


void OmtfUnpacker::produce(edm::Event& event, const edm::EventSetup& setup)
{

  bool debug = edm::MessageDrop::instance()->debugEnabled;
  theEventCounter++;
  if (debug) LogDebug ("OmtfUnpacker::produce") <<"Beginning To Unpack Event: "<<theEventCounter;

  edm::Handle<FEDRawDataCollection> allFEDRawData;
  event.getByToken(theFedDataToken,allFEDRawData);

  auto producedRPCDigis = std::make_unique<RPCDigiCollection>();
  auto producedCscLctDigis = std::make_unique<CSCCorrelatedLCTDigiCollection>();
  auto producedMuonDigis = std::make_unique<l1t::RegionalMuonCandBxCollection>(); 
  producedMuonDigis->setBXRange(-3,4);
  auto producedDTPhDigis = std::make_unique<L1MuDTChambPhContainer>();
  auto producedDTThDigis = std::make_unique<L1MuDTChambThContainer>();
  std::vector<L1MuDTChambPhDigi> phi_Container;
  std::vector<L1MuDTChambThDigi> the_Container;

  for (int fedId= 1380; fedId<= 1381; ++fedId) {

    const FEDRawData & rawData = allFEDRawData->FEDData(fedId);
    unsigned int nWords = rawData.size()/sizeof(Word64);
    LogTrace("") <<"FED : " << fedId <<" words: " << nWords;
    if (nWords==0) continue;

    //
    // FED header
    //
    const Word64* header = reinterpret_cast<const Word64* >(rawData.data());
    FEDHeader fedHeader( reinterpret_cast<const unsigned char*>(header));
    if (!fedHeader.check()) { LogTrace("") <<" ** PROBLEM **, header.check() failed, break"; break; }
    if ( fedHeader.sourceID() != fedId) {
      LogTrace("") <<" ** PROBLEM **, fedHeader.sourceID() != fedId"
          << "fedId = " << fedId<<" sourceID="<<fedHeader.sourceID();
    }
    int triggerBX = fedHeader.bxID();
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

    // 
    // FED trailer
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
      if ( fedTrailer.fragmentLength()!= nWords) {
        if (debug) LogTrace("")<<" ** PROBLEM **, fedTrailer.fragmentLength()!= nWords, break";
        break;
      }
      moreTrailers = fedTrailer.moreTrailers();
      if (debug) {
        std::ostringstream str;
        str <<" trailer: "<<  *reinterpret_cast<const std::bitset<64>*> (trailer) << std::endl;
        str <<"  trailer lenght:    "<<fedTrailer.fragmentLength()<< std::endl;
        str <<"  trailer crc:       "<<fedTrailer.crc()<< std::endl;
        str <<"  trailer evtStatus: "<<fedTrailer.evtStatus()<< std::endl;
        str <<"  trailer ttsBits:   "<<fedTrailer.ttsBits()<< std::endl;
        LogTrace("") << str.str();
      }
    }

    // 
    // dump all FED data for debug
    //
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

   
    //
    // get AMC13 payload (-> AMC's data)
    //
    amc13::Packet packetAmc13;
    if (!packetAmc13.parse( header, header+1, nWords-2, fedHeader.lvl1ID(), fedHeader.bxID(), true, false)) {
      edm::LogError("OMTF") << "Could not extract AMC13 Packet.";
      return;
    } 
    //LogTrace("") <<"AMC13 Packet: "<< packetAmc13.blocks() << " size "<<packetAmc13.size() << std::endl;  


    //
    // loop over AMC's
    //
    unsigned int blockNum=0;
    for (auto amc: packetAmc13.payload()) {
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

        unsigned int fedId = fedHeader.sourceID();
        unsigned int amcId = bh.getAMCNumber()/2+1;   // in OMTF convertsion 1-6 
        //
        // RPC data
        // 
        if (DataWord64::rpc==recordType && !theSkipRpc) {
          theRpcUnpacker.unpack(triggerBX, fedId, amcId, RpcDataWord64(*word), producedRPCDigis.get());
        }

        //
        // CSC data
        //
        if (DataWord64::csc==recordType && !theSkipCsc) {
          theCscUnpacker.unpack(fedId, amcId, CscDataWord64(*word), producedCscLctDigis.get() );
        }

        //
        // DT data
        //
        if (DataWord64::dt==recordType && !theSkipDt) {
          theDtUnpacker.unpack(fedId, amcId, DtDataWord64(*word), phi_Container, the_Container );
        }

        //
        // OMTF (muon) data
        //
        if (DataWord64::omtf==recordType && !theSkipMuon) {
          theMuonUnpacker.unpack(fedId, amcId, MuonDataWord64(*word), producedMuonDigis.get());
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

}

using namespace omtf;
DEFINE_FWK_MODULE(OmtfUnpacker);

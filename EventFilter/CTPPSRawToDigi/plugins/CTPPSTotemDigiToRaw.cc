// -*- C++ -*-
//
// Package:    temp/CTPPSTotemDigiToRaw
// Class:      CTPPSTotemDigiToRaw
// 
/**\class CTPPSTotemDigiToRaw CTPPSTotemDigiToRaw.cc temp/CTPPSTotemDigiToRaw/plugins/CTPPSTotemDigiToRaw.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Dilson De Jesus Damiao
//                   Maria Elena Pol
//         Created:  Tue, 11 Sep 2018 17:12:12 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
//#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "CondFormats/CTPPSReadoutObjects/interface/TotemDAQMapping.h"
#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"
#include "EventFilter/CTPPSRawToDigi/interface/CTPPSTotemDataFormatter.h"
#include "EventFilter/CTPPSRawToDigi/interface/VFATFrameCollection.h"
#include "CondFormats/CTPPSReadoutObjects/interface/TotemFramePosition.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// class declaration
//

class CTPPSTotemDigiToRaw : public edm::stream::EDProducer<> {
   public:
      explicit CTPPSTotemDigiToRaw(const edm::ParameterSet&);
      ~CTPPSTotemDigiToRaw();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

  unsigned long eventCounter_;
  std::set<unsigned int> fedIds_;
  int allDigiCounter_;
  int allWordCounter_;
  bool debug_;
  edm::ESWatcher<TotemReadoutRcd> recordWatcher;
  edm::EDGetTokenT<edm::DetSetVector<TotemRPDigi>> tTotemRPDigi_;
  std::map<std::map<const uint32_t, unsigned int>, std::map<short unsigned int, std::map<short unsigned int, short unsigned int>>> iDdet2fed_;
  TotemFramePosition fPos_;

      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
CTPPSTotemDigiToRaw::CTPPSTotemDigiToRaw(const edm::ParameterSet& iConfig) 
{
   //register your products
  tTotemRPDigi_ = consumes<edm::DetSetVector<TotemRPDigi> >(iConfig.getParameter<edm::InputTag>("InputLabel"));
  produces<FEDRawDataCollection>();

  // start the counters
  eventCounter_ = 0;
  allDigiCounter_ = 0;
  allWordCounter_ = 0;  
}


CTPPSTotemDigiToRaw::~CTPPSTotemDigiToRaw()
{
   edm::LogInfo("CTPPSTotemDigiToRaw")  << " CTPPSTotemDigiToRaw destructor!";
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CTPPSTotemDigiToRaw::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
  using namespace std;
  eventCounter_++;
  edm::LogInfo("CTPPSTotemDigiToRaw") << "[CTPPSTotemDigiToRaw::produce] "
    << "event number: " << eventCounter_;

  edm::Handle< edm::DetSetVector<TotemRPDigi> > digiCollection;
  iEvent.getByToken( tTotemRPDigi_, digiCollection);

  CTPPSTotemDataFormatter::RawData rawdata;
  CTPPSTotemDataFormatter::Digis digis;

  int digiCounter = 0;
  typedef vector< edm::DetSet<TotemRPDigi> >::const_iterator DI;
  for (DI di=digiCollection->begin(); di != digiCollection->end(); di++) {
    digiCounter += (di->data).size();
    digis[ di->detId()] = di->data;
  }
  allDigiCounter_ += digiCounter;
  edm::ESHandle<TotemDAQMapping> mapping;
  // label of the CTPPS sub-system
  //std::string subSystemName = "TrackingStrip";
  if (recordWatcher.check( iSetup )) {
    iSetup.get<TotemReadoutRcd>().get(mapping);
    for (const auto &p : mapping->VFATMapping) {
      //get TotemVFATInfo information
      const uint32_t pID = (p.second.symbolicID).symbolicID;
      unsigned int phwID = p.second.hwID;
      std::map<const uint32_t, unsigned int> mapSymb;
      mapSymb.insert(std::pair<const uint32_t, unsigned int>(pID,phwID));

    //get TotemFramePosition information  
      short unsigned int pFediD = p.first.getFEDId();
      fedIds_.insert(p.first.getFEDId());
      short unsigned int pIdxInFiber = p.first.getIdxInFiber();
      short unsigned int pGOHId = p.first.getGOHId();
      std::map<short unsigned int, short unsigned int> mapIdxGOH;
      mapIdxGOH.insert(std::pair<short unsigned int, short unsigned int>(pIdxInFiber,pGOHId));

      std::map<short unsigned int, std::map<short unsigned int, short unsigned int> > mapFedFiber;
      mapFedFiber.insert(std::pair<short unsigned int, std::map<short unsigned int,short unsigned int> >(pFediD,mapIdxGOH));
      iDdet2fed_.insert(std::pair<std::map<const uint32_t, unsigned int> , std::map<short unsigned int, std::map<short unsigned int, short unsigned int> > >(mapSymb,mapFedFiber));
    }
  }

  CTPPSTotemDataFormatter formatter(mapping->VFATMapping);

  formatter.formatRawData( iEvent.id().event(), rawdata, digis, iDdet2fed_);

  // create product (raw data)
  auto buffers = std::make_unique<FEDRawDataCollection>();

  // pack raw data into collection
  for (auto it = fedIds_.begin(); it != fedIds_.end(); it++) {
    FEDRawData& fedRawData = buffers->FEDData( *it );
    CTPPSTotemDataFormatter::RawData::iterator fedbuffer = rawdata.find( *it );
    if( fedbuffer != rawdata.end() ) fedRawData = fedbuffer->second;
  }
  allWordCounter_ += formatter.nWords();

  if (debug_) LogDebug("CTPPSTotemDigiToRaw")
    << "Words/Digis this iEvent: "<<digiCounter<<"(fm:"<<formatter.nDigis()<<")/"
      <<formatter.nWords()
      <<"  all: "<< allDigiCounter_ <<"/"<<allWordCounter_;

  iEvent.put(std::move(buffers));

}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
CTPPSTotemDigiToRaw::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
CTPPSTotemDigiToRaw::endStream() {
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CTPPSTotemDigiToRaw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel",edm::InputTag("RPSiDetDigitizer"));
  descriptions.add("ctppsTotemRawData", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CTPPSTotemDigiToRaw);

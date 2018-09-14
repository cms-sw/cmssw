// -*- C++ -*-
//
// Package:    CTPPSPixelDigiToRaw
// Class:      CTPPSPixelDigiToRaw
// 
/**\class CTPPSPixelDigiToRaw CTPPSPixelDigiToRaw.cc 

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Dilson De Jesus Damiao
//                   Maria Elena Pol
//         Created:  Wed, 12 Sep 2018 12:59:49 GMT
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
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "EventFilter/CTPPSRawToDigi/interface/CTPPSPixelDataFormatter.h"

#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelDAQMapping.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelFramePosition.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//
// class declaration
//

class CTPPSPixelDigiToRaw : public edm::stream::EDProducer<> {
   public:
      explicit CTPPSPixelDigiToRaw(const edm::ParameterSet&);
      ~CTPPSPixelDigiToRaw();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

      // ----------member data ---------------------------
  unsigned long eventCounter_;
  int allDigiCounter_;
  int allWordCounter_;
  bool debug_;
  std::set<unsigned int> fedIds_;
  std::string mappingLabel_;
  edm::ESWatcher<CTPPSPixelDAQMappingRcd> recordWatcher_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelDigi>> tCTPPSPixelDigi_;
  std::map<std::map<const uint32_t, std::map<short unsigned int, short unsigned int> > ,  std::map<short unsigned int, short unsigned int>  > iDdet2fed_;

  CTPPSPixelFramePosition fPos_;

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
CTPPSPixelDigiToRaw::CTPPSPixelDigiToRaw(const edm::ParameterSet& iConfig) 
{
   //register your products
 tCTPPSPixelDigi_ = consumes<edm::DetSetVector<CTPPSPixelDigi> >(iConfig.getParameter<edm::InputTag>("InputLabel"));

  // Define EDProduct type
  produces<FEDRawDataCollection>();
  mappingLabel_ = iConfig.getParameter<std::string> ("mappingLabel");
   //now do what ever other initialization is needed
  // start the counters
  eventCounter_ = 0;
  allDigiCounter_ = 0;
  allWordCounter_ = 0; 
}


CTPPSPixelDigiToRaw::~CTPPSPixelDigiToRaw()
{
   edm::LogInfo("CTPPSPixelDigiToRaw")  << " CTPPSPixelDigiToRaw destructor!";
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CTPPSPixelDigiToRaw::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

 eventCounter_++;
  edm::LogInfo("CTPPSPixelDigiToRaw") << "[CTPPSPixelDigiToRaw::produce] "
                                   << "event number: " << eventCounter_;

  edm::Handle< edm::DetSetVector<CTPPSPixelDigi> > digiCollection;
  iEvent.getByToken( tCTPPSPixelDigi_, digiCollection);

  CTPPSPixelDataFormatter::RawData rawdata;
  CTPPSPixelDataFormatter::Digis digis;
  typedef vector< edm::DetSet<CTPPSPixelDigi> >::const_iterator DI;

  int digiCounter = 0;
  for (DI di=digiCollection->begin(); di != digiCollection->end(); di++) {
    digiCounter += (di->data).size();
    digis[ di->id] = di->data;
  }
  allDigiCounter_ += digiCounter;
   edm::ESHandle<CTPPSPixelDAQMapping> mapping;
  if (recordWatcher_.check( iSetup )) {
    iSetup.get<CTPPSPixelDAQMappingRcd>().get(mapping);
    for (const auto &p : mapping->ROCMapping)    {
        const uint32_t piD = p.second.iD;
        short unsigned int pROC   = p.second.roc;
        short unsigned int pFediD = p.first.getFEDId();
        short unsigned int pFedcH = p.first.getChannelIdx();
        short unsigned int pROCcH = p.first.getROC();

        std::map<short unsigned int,short unsigned int> mapROCIdCh;
        mapROCIdCh.insert(std::pair<short unsigned int,short unsigned int>(pROC,pROCcH));

        std::map<const uint32_t, std::map<short unsigned int, short unsigned int> > mapDetRoc;
        mapDetRoc.insert(std::pair<const uint32_t, std::map<short unsigned int,short unsigned int> >(piD,mapROCIdCh));
  std::map<short unsigned int,short unsigned int> mapFedIdCh;
        mapFedIdCh.insert(std::pair<short unsigned int,short unsigned int>(pFediD,pFedcH));

        iDdet2fed_.insert(std::pair<std::map<const uint32_t, std::map<short unsigned int, short unsigned int> >, std::map<short unsigned int,short unsigned int> > (mapDetRoc,mapFedIdCh));

    }
    fedIds_ = mapping->fedIds();
  }
 CTPPSPixelDataFormatter formatter(mapping->ROCMapping);

  // create product (raw data)
  auto buffers = std::make_unique<FEDRawDataCollection>();

  // convert data to raw
  formatter.formatRawData( iEvent.id().event(), rawdata, digis, iDdet2fed_);

  // pack raw data into collection
  for (auto it = fedIds_.begin(); it != fedIds_.end(); it++) {
    FEDRawData& fedRawData = buffers->FEDData( *it );
    CTPPSPixelDataFormatter::RawData::iterator fedbuffer = rawdata.find( *it );
    if( fedbuffer != rawdata.end() ) fedRawData = fedbuffer->second;
  }
  allWordCounter_ += formatter.nWords();

  if (debug_) LogDebug("CTPPSPixelDigiToRaw")
          << "Words/Digis this iEvent: "<<digiCounter<<"(fm:"<<formatter.nDigis()<<")/"
          <<formatter.nWords()
          <<"  all: "<< allDigiCounter_ <<"/"<<allWordCounter_;

  iEvent.put(std::move(buffers));
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
CTPPSPixelDigiToRaw::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
CTPPSPixelDigiToRaw::endStream() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
CTPPSPixelDigiToRaw::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
CTPPSPixelDigiToRaw::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
CTPPSPixelDigiToRaw::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
CTPPSPixelDigiToRaw::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CTPPSPixelDigiToRaw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel",edm::InputTag("RPixDetDigitizer"));
  desc.add<std::string>("mappingLabel","RPix");
  descriptions.add("ctppsPixelRawData", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CTPPSPixelDigiToRaw);

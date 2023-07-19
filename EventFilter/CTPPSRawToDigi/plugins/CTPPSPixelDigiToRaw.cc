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
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "EventFilter/CTPPSRawToDigi/interface/CTPPSPixelDataFormatter.h"

#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelDAQMapping.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelFramePosition.h"
#include "EventFilter/CTPPSRawToDigi/interface/CTPPSPixelErrorSummary.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//
// class declaration
//

class CTPPSPixelDigiToRaw : public edm::stream::EDProducer<> {
public:
  explicit CTPPSPixelDigiToRaw(const edm::ParameterSet&);
  ~CTPPSPixelDigiToRaw() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  unsigned long eventCounter_;
  int allDigiCounter_;
  int allWordCounter_;
  bool debug_;
  std::set<unsigned int> fedIds_;
  std::string mappingLabel_;
  edm::ESWatcher<CTPPSPixelDAQMappingRcd> recordWatcher_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelDigi>> tCTPPSPixelDigi_;
  edm::ESGetToken<CTPPSPixelDAQMapping, CTPPSPixelDAQMappingRcd> tCTPPSPixelDAQMapping_;
  std::vector<CTPPSPixelDataFormatter::PPSPixelIndex> v_iDdet2fed_;
  CTPPSPixelFramePosition fPos_;
  CTPPSPixelErrorSummary eSummary_;
  bool isRun3_;
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
    : eventCounter_(0),
      allDigiCounter_(0),
      allWordCounter_(0),
      debug_(false),
      mappingLabel_(iConfig.getParameter<std::string>("mappingLabel")),
      eSummary_("CTPPSPixelDataFormatter", "[ctppsPixelRawToDigi]", false) {
  //register your products
  tCTPPSPixelDigi_ = consumes<edm::DetSetVector<CTPPSPixelDigi>>(iConfig.getParameter<edm::InputTag>("InputLabel"));
  tCTPPSPixelDAQMapping_ = esConsumes<CTPPSPixelDAQMapping, CTPPSPixelDAQMappingRcd>();

  // Define EDProduct type
  produces<FEDRawDataCollection>();

  isRun3_ = iConfig.getParameter<bool>("isRun3");
}

CTPPSPixelDigiToRaw::~CTPPSPixelDigiToRaw() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void CTPPSPixelDigiToRaw::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  eventCounter_++;

  edm::Handle<edm::DetSetVector<CTPPSPixelDigi>> digiCollection;
  iEvent.getByToken(tCTPPSPixelDigi_, digiCollection);

  CTPPSPixelDataFormatter::RawData rawdata;
  CTPPSPixelDataFormatter::Digis digis;

  int digiCounter = 0;
  for (auto const& di : *digiCollection) {
    digiCounter += (di.data).size();
    digis[di.id] = di.data;
  }
  allDigiCounter_ += digiCounter;
  edm::ESHandle<CTPPSPixelDAQMapping> mapping;

  mapping = iSetup.getHandle(tCTPPSPixelDAQMapping_);
  for (const auto& p : mapping->ROCMapping)
    v_iDdet2fed_.emplace_back(CTPPSPixelDataFormatter::PPSPixelIndex{
        p.second.iD, p.second.roc, p.first.getROC(), p.first.getFEDId(), p.first.getChannelIdx()});
  fedIds_ = mapping->fedIds();

  CTPPSPixelDataFormatter formatter(mapping->ROCMapping, eSummary_);

  // create product (raw data)
  auto buffers = std::make_unique<FEDRawDataCollection>();

  std::sort(v_iDdet2fed_.begin(), v_iDdet2fed_.end(), CTPPSPixelDataFormatter::compare);

  // convert data to raw
  formatter.formatRawData(isRun3_, iEvent.id().event(), rawdata, digis, v_iDdet2fed_);

  // pack raw data into collection
  for (auto it = fedIds_.begin(); it != fedIds_.end(); it++) {
    FEDRawData& fedRawData = buffers->FEDData(*it);
    CTPPSPixelDataFormatter::RawData::iterator fedbuffer = rawdata.find(*it);
    if (fedbuffer != rawdata.end())
      fedRawData = fedbuffer->second;
  }
  allWordCounter_ += formatter.nWords();

  if (debug_)
    LogDebug("CTPPSPixelDigiToRaw") << "Words/Digis this iEvent: " << digiCounter << "(fm:" << formatter.nDigis()
                                    << ")/" << formatter.nWords() << "  all: " << allDigiCounter_ << "/"
                                    << allWordCounter_;

  iEvent.put(std::move(buffers));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void CTPPSPixelDigiToRaw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("isRun3", true);
  desc.add<edm::InputTag>("InputLabel", edm::InputTag("RPixDetDigitizer"));
  desc.add<std::string>("mappingLabel", "RPix");
  descriptions.add("ctppsPixelRawData", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CTPPSPixelDigiToRaw);

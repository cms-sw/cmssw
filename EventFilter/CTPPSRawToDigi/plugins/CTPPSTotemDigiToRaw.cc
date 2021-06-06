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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "CondFormats/PPSObjects/interface/TotemDAQMapping.h"
#include "CondFormats/PPSObjects/interface/TotemFramePosition.h"
#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "EventFilter/CTPPSRawToDigi/interface/CTPPSTotemDataFormatter.h"
#include "EventFilter/CTPPSRawToDigi/interface/VFATFrameCollection.h"
//
// class declaration
//

class CTPPSTotemDigiToRaw : public edm::stream::EDProducer<> {
public:
  explicit CTPPSTotemDigiToRaw(const edm::ParameterSet&);
  ~CTPPSTotemDigiToRaw() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  unsigned long eventCounter_;
  std::set<unsigned int> fedIds_;
  int allDigiCounter_;
  int allWordCounter_;
  bool debug_;
  edm::ESWatcher<TotemReadoutRcd> recordWatcher_;
  edm::EDGetTokenT<edm::DetSetVector<TotemRPDigi>> tTotemRPDigi_;
  edm::ESGetToken<TotemDAQMapping, TotemReadoutRcd> tTotemDAQMapping_;
  std::vector<CTPPSTotemDataFormatter::PPSStripIndex> v_iDdet2fed_;
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
    : eventCounter_(0), allDigiCounter_(0), allWordCounter_(0) {
  //register your products
  tTotemRPDigi_ = consumes<edm::DetSetVector<TotemRPDigi>>(iConfig.getParameter<edm::InputTag>("InputLabel"));
  tTotemDAQMapping_ = esConsumes<TotemDAQMapping, TotemReadoutRcd>();
  produces<FEDRawDataCollection>();
}

CTPPSTotemDigiToRaw::~CTPPSTotemDigiToRaw() {
  edm::LogInfo("CTPPSTotemDigiToRaw") << " CTPPSTotemDigiToRaw destructor!";
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void CTPPSTotemDigiToRaw::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  eventCounter_++;

  edm::Handle<edm::DetSetVector<TotemRPDigi>> digiCollection;
  iEvent.getByToken(tTotemRPDigi_, digiCollection);

  CTPPSTotemDataFormatter::RawData rawdata;
  CTPPSTotemDataFormatter::Digis digis;

  int digiCounter = 0;
  for (auto const& di : *digiCollection) {
    digiCounter += (di.data).size();
    digis[di.detId()] = di.data;
  }
  allDigiCounter_ += digiCounter;
  edm::ESHandle<TotemDAQMapping> mapping;
  // label of the CTPPS sub-system
  if (recordWatcher_.check(iSetup)) {
    mapping = iSetup.getHandle(tTotemDAQMapping_);
    for (const auto& p : mapping->VFATMapping) {
      //get TotemVFATInfo information
      fedIds_.emplace(p.first.getFEDId());
      CTPPSTotemDataFormatter::PPSStripIndex iDdet2fed = {(p.second.symbolicID).symbolicID,
                                                          p.second.hwID,
                                                          p.first.getFEDId(),
                                                          p.first.getIdxInFiber(),
                                                          p.first.getGOHId()};
      v_iDdet2fed_.emplace_back(iDdet2fed);
    }
  }

  CTPPSTotemDataFormatter formatter(mapping->VFATMapping);

  // create product (raw data)
  auto buffers = std::make_unique<FEDRawDataCollection>();

  std::sort(v_iDdet2fed_.begin(), v_iDdet2fed_.end(), CTPPSTotemDataFormatter::compare);

  // convert data to raw
  formatter.formatRawData(iEvent.id().event(), rawdata, digis, v_iDdet2fed_);

  // pack raw data into collection
  for (auto it : fedIds_) {
    FEDRawData& fedRawData = buffers->FEDData(it);
    CTPPSTotemDataFormatter::RawData::iterator fedbuffer = rawdata.find(it);
    if (fedbuffer != rawdata.end())
      fedRawData = fedbuffer->second;
  }
  allWordCounter_ += formatter.nWords();

  if (debug_)
    LogDebug("CTPPSTotemDigiToRaw") << "Words/Digis this iEvent: " << digiCounter << "(fm:" << formatter.nDigis()
                                    << ")/" << formatter.nWords() << "  all: " << allDigiCounter_ << "/"
                                    << allWordCounter_;

  iEvent.put(std::move(buffers));
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void CTPPSTotemDigiToRaw::beginStream(edm::StreamID) {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void CTPPSTotemDigiToRaw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel", edm::InputTag("RPSiDetDigitizer"));
  descriptions.add("ctppsTotemRawData", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CTPPSTotemDigiToRaw);

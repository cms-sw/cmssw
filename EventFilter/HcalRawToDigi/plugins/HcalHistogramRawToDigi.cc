#include "EventFilter/HcalRawToDigi/plugins/HcalHistogramRawToDigi.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

HcalHistogramRawToDigi::HcalHistogramRawToDigi(edm::ParameterSet const& conf)
    : unpacker_(conf.getUntrackedParameter<int>("HcalFirstFED", FEDNumbering::MINHCALFEDID)),
      fedUnpackList_(conf.getUntrackedParameter<std::vector<int> >("FEDs")),
      firstFED_(conf.getUntrackedParameter<int>("HcalFirstFED", FEDNumbering::MINHCALFEDID)) {
  std::ostringstream ss;
  for (unsigned int i = 0; i < fedUnpackList_.size(); i++)
    ss << fedUnpackList_[i] << " ";
  edm::LogInfo("HCAL") << "HcalHistogramRawToDigi will unpack FEDs ( " << ss.str() << ")";

  tok_data_ = consumes<FEDRawDataCollection>(conf.getParameter<edm::InputTag>("InputLabel"));
  tok_dbService_ = esConsumes<HcalDbService, HcalDbRecord>();

  // products produced...
  produces<HcalHistogramDigiCollection>();
}

// Virtual destructor needed.
HcalHistogramRawToDigi::~HcalHistogramRawToDigi() {}

// Functions that gets called by framework every event
void HcalHistogramRawToDigi::produce(edm::Event& e, const edm::EventSetup& es) {
  // Step A: Get Inputs
  edm::Handle<FEDRawDataCollection> rawraw;
  e.getByToken(tok_data_, rawraw);
  // get the mapping
  edm::ESHandle<HcalDbService> pSetup = es.getHandle(tok_dbService_);
  const HcalElectronicsMap* readoutMap = pSetup->getHcalMapping();

  // Step B: Create empty output
  auto prod = std::make_unique<HcalHistogramDigiCollection>();
  std::vector<HcalHistogramDigi> digis;

  // Step C: unpack all requested FEDs
  for (std::vector<int>::const_iterator i = fedUnpackList_.begin(); i != fedUnpackList_.end(); i++) {
    const FEDRawData& fed = rawraw->FEDData(*i);

    unpacker_.unpack(fed, *readoutMap, digis);
  }

  // Step B2: encapsulate vectors in actual collections
  prod->swap_contents(digis);

  // Step D: Put outputs into event
  prod->sort();
  e.put(std::move(prod));
}

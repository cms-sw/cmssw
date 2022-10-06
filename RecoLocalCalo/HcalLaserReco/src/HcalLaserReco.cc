#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDigi/interface/HcalLaserDigi.h"
#include "RecoLocalCalo/HcalLaserReco/src/HcalLaserUnpacker.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <iostream>
#include <fstream>

class HcalLaserReco : public edm::global::EDProducer<> {
public:
  explicit HcalLaserReco(const edm::ParameterSet& ps);
  void produce(edm::StreamID, edm::Event& e, const edm::EventSetup& c) const override;

private:
  int qdctdcFed_;
  HcalLaserUnpacker unpacker_;
  edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;
};

HcalLaserReco::HcalLaserReco(edm::ParameterSet const& conf)
    : qdctdcFed_(conf.getUntrackedParameter<int>("QADCTDCFED", 8)) {
  tok_raw_ = consumes<FEDRawDataCollection>(conf.getParameter<edm::InputTag>("fedRawDataCollectionTag"));

  produces<HcalLaserDigi>();
}

// Functions that gets called by framework every event
void HcalLaserReco::produce(edm::StreamID, edm::Event& e, const edm::EventSetup&) const {
  // Step A: Get Inputs
  edm::Handle<FEDRawDataCollection> rawraw;
  e.getByToken(tok_raw_, rawraw);

  // Step B: Create empty output
  auto digi = std::make_unique<HcalLaserDigi>();

  if (qdctdcFed_ >= 0) {
    // Step C: unpack all requested FEDs
    const FEDRawData& fed = rawraw->FEDData(qdctdcFed_);
    unpacker_.unpack(fed, *digi);
  }

  // Step D: Put outputs into event
  e.put(std::move(digi));
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HcalLaserReco);

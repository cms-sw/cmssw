//STL includes
#include <memory>

//framework includes
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//other includes
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

class HGCalRawToDigiFake : public edm::global::EDProducer<> {
public:
  explicit HGCalRawToDigiFake(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  edm::EDGetTokenT<HGCalDigiCollection> tok_ee_;
  edm::EDGetTokenT<HGCalDigiCollection> tok_fh_;
  edm::EDGetTokenT<HGCalDigiCollection> tok_bh_;
};

HGCalRawToDigiFake::HGCalRawToDigiFake(const edm::ParameterSet& iConfig)
    : tok_ee_(consumes<HGCalDigiCollection>(iConfig.getParameter<edm::InputTag>("eeDigis"))),
      tok_fh_(consumes<HGCalDigiCollection>(iConfig.getParameter<edm::InputTag>("fhDigis"))),
      tok_bh_(consumes<HGCalDigiCollection>(iConfig.getParameter<edm::InputTag>("bhDigis"))) {
  produces<HGCalDigiCollection>("EE");
  produces<HGCalDigiCollection>("HEfront");
  produces<HGCalDigiCollection>("HEback");
}

void HGCalRawToDigiFake::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<HGCalDigiCollection> h_ee;
  edm::Handle<HGCalDigiCollection> h_fh;
  edm::Handle<HGCalDigiCollection> h_bh;

  iEvent.getByToken(tok_ee_, h_ee);
  iEvent.getByToken(tok_fh_, h_fh);
  iEvent.getByToken(tok_bh_, h_bh);

  auto out_ee = std::make_unique<HGCalDigiCollection>();
  if (h_ee.isValid()) {
    out_ee = std::make_unique<HGCalDigiCollection>(*(h_ee.product()));
  }
  iEvent.put(std::move(out_ee), "EE");

  auto out_fh = std::make_unique<HGCalDigiCollection>();
  if (h_fh.isValid()) {
    out_fh = std::make_unique<HGCalDigiCollection>(*(h_fh.product()));
  }
  iEvent.put(std::move(out_fh), "HEfront");

  auto out_bh = std::make_unique<HGCalDigiCollection>();
  if (h_bh.isValid()) {
    out_bh = std::make_unique<HGCalDigiCollection>(*(h_bh.product()));
  }
  iEvent.put(std::move(out_bh), "HEback");
}

void HGCalRawToDigiFake::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("eeDigis", edm::InputTag("simHGCalUnsuppressedDigis:EE"));
  desc.add<edm::InputTag>("fhDigis", edm::InputTag("simHGCalUnsuppressedDigis:HEfront"));
  desc.add<edm::InputTag>("bhDigis", edm::InputTag("simHGCalUnsuppressedDigis:HEback"));

  descriptions.add("HGCalRawToDigiFake", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalRawToDigiFake);

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

class HGCDigiConverter : public edm::global::EDProducer<> {
public:
  explicit HGCDigiConverter(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  edm::EDGetTokenT<HGCEEDigiCollection> tok_ee_;
  edm::EDGetTokenT<HGCHEDigiCollection> tok_fh_;
  edm::EDGetTokenT<HGCBHDigiCollection> tok_bh_;
};

HGCDigiConverter::HGCDigiConverter(const edm::ParameterSet& iConfig)
    : tok_ee_(consumes<HGCEEDigiCollection>(iConfig.getParameter<edm::InputTag>("eeDigis"))),
      tok_fh_(consumes<HGCHEDigiCollection>(iConfig.getParameter<edm::InputTag>("fhDigis"))),
      tok_bh_(consumes<HGCBHDigiCollection>(iConfig.getParameter<edm::InputTag>("bhDigis"))) {
  produces<HGCalDigiCollection>("EE");
  produces<HGCalDigiCollection>("HEfront");
  produces<HGCalDigiCollection>("HEback");
}

void HGCDigiConverter::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<HGCEEDigiCollection> h_ee;
  edm::Handle<HGCHEDigiCollection> h_fh;
  edm::Handle<HGCBHDigiCollection> h_bh;

  iEvent.getByToken(tok_ee_, h_ee);
  iEvent.getByToken(tok_fh_, h_fh);
  iEvent.getByToken(tok_bh_, h_bh);

  auto out_ee = std::make_unique<HGCalDigiCollection>();
  if (h_ee.isValid()) {
    for (const auto& df_ee : *h_ee) {
      HGCalDataFrame tmp(df_ee.id());
      tmp.setData(df_ee.data());
      out_ee->push_back(tmp);
    }
  }
  iEvent.put(std::move(out_ee), "EE");

  auto out_fh = std::make_unique<HGCalDigiCollection>();
  if (h_fh.isValid()) {
    for (const auto& df_fh : *h_fh) {
      HGCalDataFrame tmp(df_fh.id());
      tmp.setData(df_fh.data());
      out_fh->emplace_back(tmp);
    }
  }
  iEvent.put(std::move(out_fh), "HEfront");

  auto out_bh = std::make_unique<HGCalDigiCollection>();
  if (h_bh.isValid()) {
    for (const auto& df_bh : *h_bh) {
      HGCalDataFrame tmp(df_bh.id());
      tmp.setData(df_bh.data());
      out_bh->emplace_back(tmp);
    }
  }
  iEvent.put(std::move(out_bh), "HEback");
}

void HGCDigiConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("eeDigis", edm::InputTag("mix:HGCDigisEE"));
  desc.add<edm::InputTag>("fhDigis", edm::InputTag("mix:HGCDigisHEfront"));
  desc.add<edm::InputTag>("bhDigis", edm::InputTag("mix:HGCDigisHEback"));

  descriptions.add("HGCDigiConverter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCDigiConverter);

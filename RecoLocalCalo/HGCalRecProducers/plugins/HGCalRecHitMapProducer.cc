// user include files
#include <unordered_map>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

class HGCalRecHitMapProducer : public edm::global::EDProducer<> {
public:
  HGCalRecHitMapProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<HGCRecHitCollection> hits_ee_token_;
  const edm::EDGetTokenT<HGCRecHitCollection> hits_fh_token_;
  const edm::EDGetTokenT<HGCRecHitCollection> hits_bh_token_;
};

DEFINE_FWK_MODULE(HGCalRecHitMapProducer);

using DetIdRecHitMap = std::unordered_map<DetId, const HGCRecHit*>;

HGCalRecHitMapProducer::HGCalRecHitMapProducer(const edm::ParameterSet& ps)
    : hits_ee_token_(consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("EEInput"))),
      hits_fh_token_(consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("FHInput"))),
      hits_bh_token_(consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("BHInput"))) {
  produces<DetIdRecHitMap>();
}

void HGCalRecHitMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("EEInput", {"HGCalRecHit", "HGCEERecHits"});
  desc.add<edm::InputTag>("FHInput", {"HGCalRecHit", "HGCHEFRecHits"});
  desc.add<edm::InputTag>("BHInput", {"HGCalRecHit", "HGCHEBRecHits"});
  descriptions.add("hgcalRecHitMapProducer", desc);
}

void HGCalRecHitMapProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const {
  auto hitMap = std::make_unique<DetIdRecHitMap>();
  const auto& ee_hits = evt.get(hits_ee_token_);
  const auto& fh_hits = evt.get(hits_fh_token_);
  const auto& bh_hits = evt.get(hits_bh_token_);

  for (const auto& hit : ee_hits) {
    hitMap->emplace(hit.detid(), &hit);
  }

  for (const auto& hit : fh_hits) {
    hitMap->emplace(hit.detid(), &hit);
  }

  for (const auto& hit : bh_hits) {
    hitMap->emplace(hit.detid(), &hit);
  }
  evt.put(std::move(hitMap));
}

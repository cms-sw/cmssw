#ifndef __RecoLocalCalo_HGCRecProducers_HGCalRecHitMapProducer_H__
#define __RecoLocalCalo_HGCRecProducers_HGCalRecHitMapProducer_H__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"


class HGCalRecHitMapProducer : public edm::stream::EDProducer<> {
public:
  HGCalRecHitMapProducer(const edm::ParameterSet&);
  ~HGCalRecHitMapProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<HGCRecHitCollection> hits_ee_token;
  edm::EDGetTokenT<HGCRecHitCollection> hits_fh_token;
  edm::EDGetTokenT<HGCRecHitCollection> hits_bh_token;
};

DEFINE_FWK_MODULE(HGCalRecHitMapProducer);

HGCalRecHitMapProducer::HGCalRecHitMapProducer(const edm::ParameterSet& ps)
{
  hits_ee_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCEEInput"));
  hits_fh_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCFHInput"));
  hits_bh_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("HGCBHInput"));
  produces<std::map<DetId, HGCRecHit*>>();
}

void HGCalRecHitMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("HGCEEInput", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  desc.add<edm::InputTag>("HGCFHInput", edm::InputTag("HGCalRecHit", "HGCHEFRecHits"));
  desc.add<edm::InputTag>("HGCBHInput", edm::InputTag("HGCalRecHit", "HGCHEBRecHits"));
  descriptions.add("hgcalRecHitMapProducer", desc);
}

void HGCalRecHitMapProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  std::unique_ptr<std::map<DetId, const HGCRecHit*>> hitMap;
  edm::Handle<HGCRecHitCollection> ee_hits;
  edm::Handle<HGCRecHitCollection> fh_hits;
  edm::Handle<HGCRecHitCollection> bh_hits;


  evt.getByToken(hits_ee_token, ee_hits);
  evt.getByToken(hits_fh_token, fh_hits);
  evt.getByToken(hits_bh_token, bh_hits);
  for (const auto& hit : *ee_hits.product()) {
    hitMap->emplace(hit.detid(), &hit);
  }

  for (const auto& hit : *fh_hits.product()) {
    hitMap->emplace(hit.detid(), &hit);
  }

  for (const auto& hit : *bh_hits.product()) {
    hitMap->emplace(hit.detid(), &hit);
  }
  evt.put(std::move(hitMap));
}

#endif

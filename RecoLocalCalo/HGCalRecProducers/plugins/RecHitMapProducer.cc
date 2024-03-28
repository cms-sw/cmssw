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
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

class RecHitMapProducer : public edm::global::EDProducer<> {
public:
  RecHitMapProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<HGCRecHitCollection> hits_ee_token_;
  const edm::EDGetTokenT<HGCRecHitCollection> hits_fh_token_;
  const edm::EDGetTokenT<HGCRecHitCollection> hits_bh_token_;
  const edm::EDGetTokenT<reco::PFRecHitCollection> hits_eb_token_;
  const edm::EDGetTokenT<reco::PFRecHitCollection> hits_hb_token_;
  const edm::EDGetTokenT<reco::PFRecHitCollection> hits_ho_token_;
  bool hgcalOnly_;
};

DEFINE_FWK_MODULE(RecHitMapProducer);

using DetIdRecHitMap = std::unordered_map<DetId, const unsigned int>;

RecHitMapProducer::RecHitMapProducer(const edm::ParameterSet& ps)
    : hits_ee_token_(consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("EEInput"))),
      hits_fh_token_(consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("FHInput"))),
      hits_bh_token_(consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("BHInput"))),
      hits_eb_token_(consumes<reco::PFRecHitCollection>(ps.getParameter<edm::InputTag>("EBInput"))),
      hits_hb_token_(consumes<reco::PFRecHitCollection>(ps.getParameter<edm::InputTag>("HBInput"))),
      hits_ho_token_(consumes<reco::PFRecHitCollection>(ps.getParameter<edm::InputTag>("HOInput"))),
      hgcalOnly_(ps.getParameter<bool>("hgcalOnly")) {
  produces<DetIdRecHitMap>("hgcalRecHitMap");
  if (!hgcalOnly_)
    produces<DetIdRecHitMap>("barrelRecHitMap");
}

void RecHitMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("EEInput", {"HGCalRecHit", "HGCEERecHits"});
  desc.add<edm::InputTag>("FHInput", {"HGCalRecHit", "HGCHEFRecHits"});
  desc.add<edm::InputTag>("BHInput", {"HGCalRecHit", "HGCHEBRecHits"});
  desc.add<edm::InputTag>("EBInput", {"particleFlowRecHitECAL", ""});
  desc.add<edm::InputTag>("HBInput", {"particleFlowRecHitHBHE", ""});
  desc.add<edm::InputTag>("HOInput", {"particleFlowRecHitHO", ""});
  desc.add<bool>("hgcalOnly", true);
  descriptions.add("recHitMapProducer", desc);
}

void RecHitMapProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const {
  auto hitMapHGCal = std::make_unique<DetIdRecHitMap>();
  const auto& ee_hits = evt.get(hits_ee_token_);
  const auto& fh_hits = evt.get(hits_fh_token_);
  const auto& bh_hits = evt.get(hits_bh_token_);

  for (unsigned int i = 0; i < ee_hits.size(); ++i) {
    hitMapHGCal->emplace(ee_hits[i].detid(), i);
  }
  auto size = ee_hits.size();

  for (unsigned int i = 0; i < fh_hits.size(); ++i) {
    hitMapHGCal->emplace(fh_hits[i].detid(), i + size);
  }
  size += fh_hits.size();

  for (unsigned int i = 0; i < bh_hits.size(); ++i) {
    hitMapHGCal->emplace(bh_hits[i].detid(), i + size);
  }

  evt.put(std::move(hitMapHGCal), "hgcalRecHitMap");

  if (!hgcalOnly_) {
    auto hitMapBarrel = std::make_unique<DetIdRecHitMap>();
    const auto& eb_hits = evt.get(hits_eb_token_);
    const auto& hb_hits = evt.get(hits_hb_token_);
    const auto& ho_hits = evt.get(hits_ho_token_);
    size = 0;

    for (unsigned int i = 0; i < eb_hits.size(); ++i) {
      hitMapBarrel->emplace(eb_hits[i].detId(), i);
    }
    size += eb_hits.size();

    for (unsigned int i = 0; i < hb_hits.size(); ++i) {
      hitMapBarrel->emplace(hb_hits[i].detId(), i + size);
    }
    size += hb_hits.size();

    for (unsigned int i = 0; i < ho_hits.size(); ++i) {
      hitMapBarrel->emplace(ho_hits[i].detId(), i + size);
    }
    evt.put(std::move(hitMapBarrel), "barrelRecHitMap");
  }
}

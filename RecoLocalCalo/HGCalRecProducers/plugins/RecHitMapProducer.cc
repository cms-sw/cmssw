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
#include "CommonTools/RecoAlgos/interface/MultiVectorManager.h"

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

  // Retrieve collections
  const auto& ee_hits = evt.getHandle(hits_ee_token_);
  const auto& fh_hits = evt.getHandle(hits_fh_token_);
  const auto& bh_hits = evt.getHandle(hits_bh_token_);

  // Check validity of all handles
  if (!ee_hits.isValid() || !fh_hits.isValid() || !bh_hits.isValid()) {
    edm::LogWarning("HGCalRecHitMapProducer") << "One or more hit collections are unavailable. Returning an empty map.";
    evt.put(std::move(hitMapHGCal), "hgcalRecHitMap");
    return;
  }

  MultiVectorManager<HGCRecHit> rechitManager;
  rechitManager.addVector(*ee_hits);
  rechitManager.addVector(*fh_hits);
  rechitManager.addVector(*bh_hits);

  for (unsigned int i = 0; i < rechitManager.size(); ++i) {
    const auto recHitDetId = rechitManager[i].detid();
    hitMapHGCal->emplace(recHitDetId, i);
  }

  evt.put(std::move(hitMapHGCal), "hgcalRecHitMap");

  if (!hgcalOnly_) {
    auto hitMapBarrel = std::make_unique<DetIdRecHitMap>();
    MultiVectorManager<reco::PFRecHit> barrelRechitManager;
    barrelRechitManager.addVector(evt.get(hits_eb_token_));
    barrelRechitManager.addVector(evt.get(hits_hb_token_));
    barrelRechitManager.addVector(evt.get(hits_ho_token_));
    for (unsigned int i = 0; i < barrelRechitManager.size(); ++i) {
      const auto recHitDetId = barrelRechitManager[i].detId();
      hitMapBarrel->emplace(recHitDetId, i);
    }
    evt.put(std::move(hitMapBarrel), "barrelRecHitMap");
  }
}

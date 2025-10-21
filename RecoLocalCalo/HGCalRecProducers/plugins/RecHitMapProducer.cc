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
#include "DataFormats/Common/interface/MultiSpan.h"

class RecHitMapProducer : public edm::global::EDProducer<> {
public:
  RecHitMapProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hgcal_hits_token_;
  std::vector<edm::EDGetTokenT<reco::PFRecHitCollection>> barrel_hits_token_;

  bool hgcalOnly_;
};

DEFINE_FWK_MODULE(RecHitMapProducer);

using DetIdRecHitMap = std::unordered_map<DetId, const unsigned int>;

RecHitMapProducer::RecHitMapProducer(const edm::ParameterSet& ps) : hgcalOnly_(ps.getParameter<bool>("hgcalOnly")) {
  std::vector<edm::InputTag> tags = ps.getParameter<std::vector<edm::InputTag>>("hits");
  for (auto& tag : tags) {
    if (tag.label().find("HGCalRecHit") != std::string::npos) {
      hgcal_hits_token_.push_back(consumes<HGCRecHitCollection>(tag));
    } else {
      barrel_hits_token_.push_back(consumes<reco::PFRecHitCollection>(tag));
    }
  }

  produces<DetIdRecHitMap>("hgcalRecHitMap");
  if (!hgcalOnly_)
    produces<DetIdRecHitMap>("barrelRecHitMap");
}

void RecHitMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("hits",
                                       {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
  desc.add<bool>("hgcalOnly", true);
  descriptions.add("recHitMapProducer", desc);
}

void RecHitMapProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const {
  auto hitMapHGCal = std::make_unique<DetIdRecHitMap>();

  // Retrieve collections
  assert(hgcal_hits_token_.size() == 3);
  const auto& ee_hits = evt.getHandle(hgcal_hits_token_[0]);
  const auto& fh_hits = evt.getHandle(hgcal_hits_token_[1]);
  const auto& bh_hits = evt.getHandle(hgcal_hits_token_[2]);

  // Check validity of all handles
  if (!ee_hits.isValid() || !fh_hits.isValid() || !bh_hits.isValid()) {
    edm::LogWarning("HGCalRecHitMapProducer")
        << "One or more HGCal hit collections are unavailable. Returning an empty map.";
    evt.put(std::move(hitMapHGCal), "hgcalRecHitMap");
  } else {
    // TODO may be worth to avoid dependency on the order
    // of the collections, maybe using a map
    edm::MultiSpan<HGCRecHit> rechitSpan;
    rechitSpan.add(*ee_hits);
    rechitSpan.add(*fh_hits);
    rechitSpan.add(*bh_hits);

    for (unsigned int i = 0; i < rechitSpan.size(); ++i) {
      const auto recHitDetId = rechitSpan[i].detid();
      hitMapHGCal->emplace(recHitDetId, i);
    }

    evt.put(std::move(hitMapHGCal), "hgcalRecHitMap");
  }

  if (hgcalOnly_) {
    return;
  }

  auto hitMapBarrel = std::make_unique<DetIdRecHitMap>();

  assert(barrel_hits_token_.size() == 2);
  const auto& ecal_hits = evt.getHandle(barrel_hits_token_[0]);
  const auto& hbhe_hits = evt.getHandle(barrel_hits_token_[1]);

  if (!ecal_hits.isValid() || !hbhe_hits.isValid()) {
    edm::LogWarning("HGCalRecHitMapProducer")
        << "One or more barrel hit collections are unavailable. Returning an empty map.";
    evt.put(std::move(hitMapBarrel), "barrelRecHitMap");
  } else {
    edm::MultiSpan<reco::PFRecHit> barrelRechitSpan;
    barrelRechitSpan.add(evt.get(barrel_hits_token_[0]));
    barrelRechitSpan.add(evt.get(barrel_hits_token_[1]));
    for (unsigned int i = 0; i < barrelRechitSpan.size(); ++i) {
      const auto recHitDetId = barrelRechitSpan[i].detId();
      hitMapBarrel->emplace(recHitDetId, i);
    }

    evt.put(std::move(hitMapBarrel), "barrelRecHitMap");
  }
}

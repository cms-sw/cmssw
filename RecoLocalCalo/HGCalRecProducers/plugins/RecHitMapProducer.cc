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
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/Common/interface/RefProdVector.h"
#include "DataFormats/Common/interface/MultiSpan.h"

class RecHitMapProducer : public edm::global::EDProducer<> {
public:
  RecHitMapProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  std::vector<edm::InputTag> hgcal_hits_tag_;
  std::vector<edm::InputTag> pf_hits_tag_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hgcal_hits_token_;
  std::vector<edm::EDGetTokenT<reco::PFRecHitCollection>> pf_hits_token_;

  bool doHgcalHits_;
  bool doPFHits_;
};

DEFINE_FWK_MODULE(RecHitMapProducer);

using DetIdRecHitMap = std::unordered_map<DetId, const unsigned int>;

RecHitMapProducer::RecHitMapProducer(const edm::ParameterSet& ps)
    : doHgcalHits_(ps.getParameter<bool>("doHgcalHits")), doPFHits_(ps.getParameter<bool>("doPFHits")) {
  std::vector<edm::InputTag> tags = ps.getParameter<std::vector<edm::InputTag>>("hits");
  for (auto& tag : tags) {
    if (tag.label().find("HGCalRecHit") != std::string::npos) {
      hgcal_hits_tag_.push_back(tag);
      hgcal_hits_token_.push_back(consumes<HGCRecHitCollection>(tag));
    } else {
      pf_hits_tag_.push_back(tag);
      pf_hits_token_.push_back(consumes<reco::PFRecHitCollection>(tag));
    }
  }
  produces<edm::RefProdVector<HGCRecHitCollection>>("RefProdVectorHGCRecHitCollection");
  produces<DetIdRecHitMap>("hgcalRecHitMap");
  produces<edm::RefProdVector<reco::PFRecHitCollection>>("RefProdVectorPFRecHitCollection");
  produces<DetIdRecHitMap>("pfRecHitMap");
}

void RecHitMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("hits",
                                       {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
  desc.add<bool>("doHgcalHits", true);
  desc.add<bool>("doPFHits", false);
  descriptions.add("recHitMapProducer", desc);
}

void RecHitMapProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const {
  if (doHgcalHits_) {
    auto hitMapHGCal = std::make_unique<DetIdRecHitMap>();

    // Retrieve collections
    assert(hgcal_hits_token_.size() == 3);
    const auto& ee_hits = evt.getHandle(hgcal_hits_token_[0]);
    const auto& fh_hits = evt.getHandle(hgcal_hits_token_[1]);
    const auto& bh_hits = evt.getHandle(hgcal_hits_token_[2]);

    // Check validity of all handles
    if (!ee_hits.isValid() || !fh_hits.isValid() || !bh_hits.isValid()) {
      edm::LogWarning("HGCalRecHitMapProducer")
          << "One or more of the following HGCal hit collections are unavailable: ";
      for (auto& tag : hgcal_hits_tag_) {
        edm::LogWarning("HGCalRecHitMapProducer") << " - " << tag;
      }
      edm::LogWarning("HGCalRecHitMapProducer")
          << "Returning an empty map and an empty RefProdVectorHGCRecHitCollection";
      evt.put(std::make_unique<edm::RefProdVector<HGCRecHitCollection>>(), "RefProdVectorHGCRecHitCollection");
      evt.put(std::move(hitMapHGCal), "hgcalRecHitMap");
    } else {
      // Fix order by storing a edm::RefProdVector<HGCRecHitCollection>
      auto mcHGCRecHit = std::make_unique<edm::RefProdVector<HGCRecHitCollection>>();
      mcHGCRecHit->push_back(edm::RefProd<HGCRecHitCollection>(ee_hits));
      mcHGCRecHit->push_back(edm::RefProd<HGCRecHitCollection>(fh_hits));
      mcHGCRecHit->push_back(edm::RefProd<HGCRecHitCollection>(bh_hits));

      edm::MultiSpan<HGCRecHit> rechitSpan(*mcHGCRecHit);
      for (unsigned int i = 0; i < rechitSpan.size(); ++i) {
        const auto recHitDetId = rechitSpan[i].detid();
        hitMapHGCal->emplace(recHitDetId, i);
      }

      evt.put(std::move(mcHGCRecHit), "RefProdVectorHGCRecHitCollection");
      evt.put(std::move(hitMapHGCal), "hgcalRecHitMap");
    }
  } else {
    evt.put(std::make_unique<edm::RefProdVector<HGCRecHitCollection>>(), "RefProdVectorHGCRecHitCollection");
    evt.put(std::make_unique<DetIdRecHitMap>(), "hgcalRecHitMap");
  }

  if (doPFHits_) {
    auto hitMapPF = std::make_unique<DetIdRecHitMap>();

    assert(pf_hits_token_.size() == 2);
    const auto& ecal_hits = evt.getHandle(pf_hits_token_[0]);
    const auto& hbhe_hits = evt.getHandle(pf_hits_token_[1]);

    if (!ecal_hits.isValid() || !hbhe_hits.isValid()) {
      edm::LogWarning("HGCalRecHitMapProducer") << "One or more of the following PF hit collections are unavailable: ";
      for (auto& tag : pf_hits_tag_) {
        edm::LogWarning("HGCalRecHitMapProducer") << " - " << tag;
      }
      edm::LogWarning("HGCalRecHitMapProducer") << "Returning an empty map.";
      evt.put(std::make_unique<edm::RefProdVector<reco::PFRecHitCollection>>(), "RefProdVectorPFRecHitCollection");
      evt.put(std::move(hitMapPF), "pfRecHitMap");
    } else {
      // Fix order by storing a edm::RefProdVector<PFRecHitCollection>
      auto mcPFRecHit = std::make_unique<edm::RefProdVector<reco::PFRecHitCollection>>();
      mcPFRecHit->push_back(edm::RefProd<reco::PFRecHitCollection>(ecal_hits));
      mcPFRecHit->push_back(edm::RefProd<reco::PFRecHitCollection>(hbhe_hits));

      edm::MultiSpan<reco::PFRecHit> pfRechitSpan(*mcPFRecHit);
      for (unsigned int i = 0; i < pfRechitSpan.size(); ++i) {
        const auto recHitDetId = pfRechitSpan[i].detId();
        hitMapPF->emplace(recHitDetId, i);
      }

      evt.put(std::move(mcPFRecHit), "RefProdVectorPFRecHitCollection");
      evt.put(std::move(hitMapPF), "pfRecHitMap");
    }
  } else {
    evt.put(std::make_unique<edm::RefProdVector<reco::PFRecHitCollection>>(), "RefProdVectorPFRecHitCollection");
    evt.put(std::make_unique<DetIdRecHitMap>(), "pfRecHitMap");
  }
}

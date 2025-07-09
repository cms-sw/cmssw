#include <unordered_map>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "CommonTools/RecoAlgos/interface/MultiVectorManager.h"

class RecHitMapProducer : public edm::global::EDProducer<> {
public:
  RecHitMapProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  using DetIdRecHitMap = std::unordered_map<DetId, const unsigned int>;

  std::unordered_map<std::string, edm::EDGetTokenT<HGCRecHitCollection>> hgcal_hits_tokens_;
  std::unordered_map<std::string, edm::EDGetTokenT<reco::PFRecHitCollection>> barrel_hits_tokens_;

  bool hgcalOnly_;
};

DEFINE_FWK_MODULE(RecHitMapProducer);

RecHitMapProducer::RecHitMapProducer(const edm::ParameterSet& ps) : hgcalOnly_(ps.getParameter<bool>("hgcalOnly")) {
  const edm::ParameterSet& hitPSet = ps.getParameter<edm::ParameterSet>("hits");
  for (const auto& entry : hitPSet.getParameterNamesForType<edm::InputTag>()) {
    edm::InputTag tag = hitPSet.getParameter<edm::InputTag>(entry);
    if (tag.label().find("HGCalRecHit") != std::string::npos) {
      hgcal_hits_tokens_.emplace(entry, consumes<HGCRecHitCollection>(tag));
    } else {
      barrel_hits_tokens_.emplace(entry, consumes<reco::PFRecHitCollection>(tag));
    }
  }

  produces<DetIdRecHitMap>("hgcalRecHitMap");
  if (!hgcalOnly_) {
    produces<DetIdRecHitMap>("barrelRecHitMap");
  }
}

void RecHitMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription hitDesc;
  hitDesc.add<edm::InputTag>("HGCEE", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  hitDesc.add<edm::InputTag>("HGCHEF", edm::InputTag("HGCalRecHit", "HGCHEFRecHits"));
  hitDesc.add<edm::InputTag>("HGCHEB", edm::InputTag("HGCalRecHit", "HGCHEBRecHits"));
  // Optionally allow barrel collections to be defined
  hitDesc.addOptional<edm::InputTag>("ECAL", edm::InputTag("particleFlowRecHitECAL"));
  hitDesc.addOptional<edm::InputTag>("HCAL", edm::InputTag("particleFlowRecHitHBHE"));

  edm::ParameterSetDescription desc;
  desc.add<edm::ParameterSetDescription>("hits", hitDesc);
  desc.add<bool>("hgcalOnly", true);
  descriptions.add("recHitMapProducer", desc);
}

void RecHitMapProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup&) const {
  auto hitMapHGCal = std::make_unique<DetIdRecHitMap>();
  MultiVectorManager<HGCRecHit> rechitManager;

  for (const auto& [name, token] : hgcal_hits_tokens_) {
    const auto& handle = evt.getHandle(token);
    if (handle.isValid()) {
      rechitManager.addVector(*handle);
    } else {
      edm::LogWarning("RecHitMapProducer") << "HGCal collection \"" << name << "\" is invalid.";
    }
  }

  for (unsigned int i = 0; i < rechitManager.size(); ++i) {
    hitMapHGCal->emplace(rechitManager[i].detid(), i);
  }

  evt.put(std::move(hitMapHGCal), "hgcalRecHitMap");

  if (hgcalOnly_)
    return;

  auto hitMapBarrel = std::make_unique<DetIdRecHitMap>();
  MultiVectorManager<reco::PFRecHit> barrelManager;

  for (const auto& [name, token] : barrel_hits_tokens_) {
    const auto& handle = evt.getHandle(token);
    if (handle.isValid()) {
      barrelManager.addVector(*handle);
    } else {
      edm::LogWarning("RecHitMapProducer") << "Barrel collection \"" << name << "\" is invalid.";
    }
  }

  for (unsigned int i = 0; i < barrelManager.size(); ++i) {
    hitMapBarrel->emplace(barrelManager[i].detId(), i);
  }

  evt.put(std::move(hitMapBarrel), "barrelRecHitMap");
}

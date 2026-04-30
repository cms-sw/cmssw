// system includes
#include <memory>

// framework stuff
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

// data formats
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

class FastTrackerRecHitMaskFromSeedProducer : public edm::global::EDProducer<> {
public:
  explicit FastTrackerRecHitMaskFromSeedProducer(const edm::ParameterSet&);

  ~FastTrackerRecHitMaskFromSeedProducer() override {}

  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // an alias
  using QualityMaskCollection = std::vector<unsigned char>;

  // tokens
  edm::EDGetTokenT<TrajectorySeedCollection> trajectories_;

  edm::EDGetTokenT<FastTrackerRecHitCollection> recHits_;

  edm::EDGetTokenT<std::vector<bool> > oldHitMaskToken_;

  edm::EDPutTokenT<std::vector<bool> > collectedHits_;
};

FastTrackerRecHitMaskFromSeedProducer::FastTrackerRecHitMaskFromSeedProducer(const edm::ParameterSet& iConfig)
    : trajectories_(consumes(iConfig.getParameter<edm::InputTag>("trajectories"))),
      recHits_(consumes(iConfig.getParameter<edm::InputTag>("recHits"))),
      collectedHits_(produces()) {
  auto const& oldHitRemovalInfo = iConfig.getParameter<edm::InputTag>("oldHitRemovalInfo");
  if (!oldHitRemovalInfo.label().empty()) {
    oldHitMaskToken_ = consumes<std::vector<bool> >(oldHitRemovalInfo);
  }
}

void FastTrackerRecHitMaskFromSeedProducer::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  // the product

  std::vector<bool> collectedHits;

  // input

  auto const& recHits = iEvent.get(recHits_);

  if (!oldHitMaskToken_.isUninitialized()) {
    auto const& oldHitMasks = iEvent.get(oldHitMaskToken_);
    collectedHits.insert(collectedHits.begin(), oldHitMasks.begin(), oldHitMasks.end());
  }
  collectedHits.resize(recHits.size(), false);

  auto const& seeds = iEvent.get(trajectories_);

  // loop over seed tracks and mask hits
  for (auto const& seed : seeds) {
    for (auto const& hit : seed.recHits()) {
      if (!hit.isValid())
        continue;
      const FastTrackerRecHit& fasthit = static_cast<const FastTrackerRecHit&>(hit);
      // note: for matched hits nIds() returns 2, otherwise 1
      for (unsigned id_index = 0; id_index < fasthit.nIds(); id_index++) {
        collectedHits[unsigned(fasthit.id(id_index))] = true;
      }
    }
  }

  iEvent.emplace(collectedHits_, std::move(collectedHits));
}

void FastTrackerRecHitMaskFromSeedProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("trajectories", edm::InputTag("initialStepSeeds"));
  desc.add<edm::InputTag>("recHits", edm::InputTag("fastTrackerRecHits"));
  desc.add<edm::InputTag>("oldHitRemovalInfo", edm::InputTag(""));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(FastTrackerRecHitMaskFromSeedProducer);

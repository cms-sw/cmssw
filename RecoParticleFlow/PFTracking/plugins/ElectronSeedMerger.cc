// -*- C++ -*-
//
// Package:    PFTracking
// Class:      ElectronSeedMerger
//
// Original Author:  Michele Pioppi

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

class ElectronSeedMerger : public edm::global::EDProducer<> {
public:
  explicit ElectronSeedMerger(const edm::ParameterSet&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  const edm::EDGetTokenT<reco::ElectronSeedCollection> ecalSeedToken_;
  edm::EDGetTokenT<reco::ElectronSeedCollection> tkSeedToken_;
};

using namespace edm;
using namespace std;
using namespace reco;

ElectronSeedMerger::ElectronSeedMerger(const ParameterSet& iConfig)
    : ecalSeedToken_{consumes<ElectronSeedCollection>(iConfig.getParameter<InputTag>("EcalBasedSeeds"))} {
  edm::InputTag tkSeedLabel_ = iConfig.getParameter<InputTag>("TkBasedSeeds");
  if (!tkSeedLabel_.label().empty())
    tkSeedToken_ = consumes<ElectronSeedCollection>(tkSeedLabel_);

  produces<ElectronSeedCollection>();
}

// ------------ method called to produce the data  ------------
void ElectronSeedMerger::produce(edm::StreamID, Event& iEvent, const EventSetup& iSetup) const {
  //HANDLE THE INPUT SEED COLLECTIONS
  auto const& eSeeds = iEvent.get(ecalSeedToken_);

  ElectronSeedCollection tSeedsEmpty;
  auto const& tSeeds = tkSeedToken_.isUninitialized() ? tSeedsEmpty : iEvent.get(tkSeedToken_);

  //CREATE OUTPUT COLLECTION
  auto output = std::make_unique<ElectronSeedCollection>();
  output->reserve(eSeeds.size() + tSeeds.size());

  //VECTOR FOR MATCHED SEEDS
  vector<bool> tSeedsMatched(tSeeds.size(), false);

  //LOOP OVER THE ECAL SEED COLLECTION
  for (auto newSeed : eSeeds) {  //  make a copy

    //LOOP OVER THE TK SEED COLLECTION
    int it = -1;
    for (auto const& tSeed : tSeeds) {
      it++;

      //HITS FOR TK SEED
      unsigned int hitShared = 0;
      unsigned int hitSeed = 0;
      for (auto const& eh : newSeed.recHits()) {
        if (!eh.isValid())
          continue;
        hitSeed++;

        for (auto const& th : tSeed.recHits()) {
          if (!th.isValid())
            continue;
          //CHECK THE HIT COMPATIBILITY
          if (eh.sharesInput(&th, TrackingRecHit::all)) {
            hitShared++;
            break;
          }
        }
      }
      if (hitShared == hitSeed) {
        tSeedsMatched[it] = true;
        newSeed.setCtfTrack(tSeed.ctfTrack());
        break;
      }
      if (hitShared == (hitSeed - 1)) {
        newSeed.setCtfTrack(tSeed.ctfTrack());
      } else if ((hitShared > 0 || tSeed.nHits() == 0) && !newSeed.isTrackerDriven()) {
        //try to find hits in the full track
        unsigned int hitSharedOnTrack = 0;
        for (auto const& eh : newSeed.recHits()) {
          if (!eh.isValid())
            continue;
          for (auto const* th : tSeed.ctfTrack()->recHits()) {
            if (!th->isValid())
              continue;
            // hits on tracks are not matched : use ::some
            if (eh.sharesInput(th, TrackingRecHit::some)) {
              hitSharedOnTrack++;
              break;
            }
          }
        }
        if (hitSharedOnTrack == hitSeed) {
          tSeedsMatched[it] = true;
          newSeed.setCtfTrack(tSeed.ctfTrack());
          break;
        }
        if (hitSharedOnTrack == (hitSeed - 1)) {
          newSeed.setCtfTrack(tSeed.ctfTrack());
        }
      }
    }

    output->push_back(newSeed);
  }

  //FILL THE COLLECTION WITH UNMATCHED TK-BASED SEED
  for (unsigned int it = 0; it < tSeeds.size(); it++) {
    if (!tSeedsMatched[it])
      output->push_back(tSeeds[it]);
  }

  //PUT THE MERGED COLLECTION IN THE EVENT
  iEvent.put(std::move(output));
}

DEFINE_FWK_MODULE(ElectronSeedMerger);

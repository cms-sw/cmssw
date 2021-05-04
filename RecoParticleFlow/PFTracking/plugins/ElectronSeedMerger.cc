// -*- C++ -*-
//
// Package:    PFTracking
// Class:      ElectronSeedMerger
//
// Original Author:  Michele Pioppi

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
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
  //CREATE OUTPUT COLLECTION
  auto output = std::make_unique<ElectronSeedCollection>();

  //HANDLE THE INPUT SEED COLLECTIONS
  Handle<ElectronSeedCollection> EcalBasedSeeds;
  iEvent.getByToken(ecalSeedToken_, EcalBasedSeeds);
  ElectronSeedCollection ESeed = *(EcalBasedSeeds.product());

  Handle<ElectronSeedCollection> TkBasedSeeds;
  ElectronSeedCollection TSeed;
  if (!tkSeedToken_.isUninitialized()) {
    iEvent.getByToken(tkSeedToken_, TkBasedSeeds);
    TSeed = *(TkBasedSeeds.product());
  }

  //VECTOR FOR MATCHED SEEDS
  vector<bool> TSeedMatched;
  for (unsigned int it = 0; it < TSeed.size(); it++) {
    TSeedMatched.push_back(false);
  }

  //LOOP OVER THE ECAL SEED COLLECTION
  ElectronSeedCollection::const_iterator e_beg = ESeed.begin();
  ElectronSeedCollection::const_iterator e_end = ESeed.end();
  for (; e_beg != e_end; ++e_beg) {
    ElectronSeed NewSeed = *(e_beg);
    bool AlreadyMatched = false;

    //LOOP OVER THE TK SEED COLLECTION
    for (unsigned int it = 0; it < TSeed.size(); it++) {
      if (AlreadyMatched)
        continue;

      //HITS FOR TK SEED
      unsigned int hitShared = 0;
      unsigned int hitSeed = 0;
      for (auto const& eh : e_beg->recHits()) {
        if (!eh.isValid())
          continue;
        hitSeed++;
        bool Shared = false;
        for (auto const& th : TSeed[it].recHits()) {
          if (!th.isValid())
            continue;
          //CHECK THE HIT COMPATIBILITY: put back sharesInput
          // as soon Egamma solves the bug on the seed collection
          if (eh.sharesInput(&th, TrackingRecHit::all))
            Shared = true;
          //   if(eh->geographicalId() == th->geographicalId() &&
          // 	     (eh->localPosition() - th->localPosition()).mag() < 0.001) Shared=true;
        }
        if (Shared)
          hitShared++;
      }
      if (hitShared == hitSeed) {
        AlreadyMatched = true;
        TSeedMatched[it] = true;
        NewSeed.setCtfTrack(TSeed[it].ctfTrack());
      }
      if (hitShared == (hitSeed - 1)) {
        NewSeed.setCtfTrack(TSeed[it].ctfTrack());
      }
    }

    output->push_back(NewSeed);
  }

  //FILL THE COLLECTION WITH UNMATCHED TK-BASED SEED
  for (unsigned int it = 0; it < TSeed.size(); it++) {
    if (!TSeedMatched[it])
      output->push_back(TSeed[it]);
  }

  //PUT THE MERGED COLLECTION IN THE EVENT
  iEvent.put(std::move(output));
}

DEFINE_FWK_MODULE(ElectronSeedMerger);

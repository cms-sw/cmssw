
#include "DataFormats/CSCRecHit/test/CSCSharesInputTest.h"

#include <string>
#include <sstream>
#include <iostream>

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"

CSCSharesInputTest::CSCSharesInputTest(const edm::ParameterSet &myConfig) {
  rh_token = consumes<CSCRecHit2DCollection>(myConfig.getParameter<edm::InputTag>("CSCRecHitCollection"));
  mu_token = consumes<edm::View<reco::Muon> >(myConfig.getParameter<edm::InputTag>("MuonCollection"));

  // Set up plots, ntuples
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> rootFile;

  ntuples_["perEvent"] = rootFile->make<TNtuple>(
      "perEvent",
      "Interesting data on a per-event basis",
      "Muons:RecHits:TrackingRecHits:AllMatchedRecHits:SomeMatchedRecHits:AllWiresMatchedRecHits:"
      "SomeWiresMatchedRecHits:AllStripsMatchedRecHits:SomeStripsMatchedRecHits:NotMatchedRecHits");
  ntuples_["perMuon"] = rootFile->make<TNtuple>(
      "perMuon",
      "Interesting data on a per-muon basis",
      "TrackingRecHits:AllMatchedRecHits:SomeMatchedRecHits:AllWiresMatchedRecHits:SomeWiresMatchedRecHits:"
      "AllStripsMatchedRecHits:SomeStripsMatchedRecHits:NotMatchedRecHits");
  ntuples_["perRecHit"] = rootFile->make<TNtuple>(
      "perRecHit",
      "Interesting data per RecHit",
      "AllMatched:SomeMatched:AllWiresMatched:SomeWiresMatched:AllStripsMatched:SomeStripsMatched:NotMatched");
}

void CSCSharesInputTest::beginJob() {}

void CSCSharesInputTest::analyze(const edm::Event &myEvent, const edm::EventSetup &mySetup) {
  ++counts_["Events"];  // presuming default init of int elem to zero

  edm::Handle<edm::View<reco::Muon> > muonHandle;
  myEvent.getByToken(mu_token, muonHandle);

  edm::View<reco::Muon> muons = *muonHandle;

  edm::Handle<CSCRecHit2DCollection> recHits;
  myEvent.getByToken(rh_token, recHits);

  // Muons:RecHits:TrackingRecHits:AllMatchedRecHits:SomeMatchedRecHits:AllWiresMatchedRecHits:SomeWiresMatchedRecHits:AllStripsMatchedRecHits:SomeStripsMatchedRecHits:NotMatchedRecHits
  float perEventData[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  counts_["Muons"] += muons.size();
  perEventData[0] = muons.size();
  counts_["RecHits"] += recHits->size();
  perEventData[1] = recHits->size();

  for (edm::View<reco::Muon>::const_iterator iMuon = muons.begin(); iMuon != muons.end(); ++iMuon) {
    // TrackingRecHits:AllMatchedRecHits:SomeMatchedRecHits:AllWiresMatchedRecHits:SomeWiresMatchedRecHits:AllStripsMatchedRecHits:SomeStripsMatchedRecHits:NotMatchedRecHits
    float perMuonData[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // note:  only outer tracks are considered
    reco::Track *track = new reco::Track(*(iMuon->outerTrack()));

    counts_["TrackingRecHits"] += track->recHitsSize();
    perEventData[2] += track->recHitsSize();
    perMuonData[0] += track->recHitsSize();

    for (auto const &jHit : track->recHits()) {
      // AllMatched:SomeMatched:AllWiresMatched:SomeWiresMatched:AllStripsMatched:SomeStripsMatched:NotMatched
      float perRecHitData[7] = {0, 0, 0, 0, 0, 0, 0};

      // Kill us quickly if this is not a CSCRecHit.  Also allows us to use the CSCRecHit version of sharesInput, which we like.
      const CSCRecHit2D *myHit = dynamic_cast<const CSCRecHit2D *>(jHit);
      if (myHit == 0) {
        ++counts_["NotMatchedRecHits"];
        ++perEventData[9];
        ++perMuonData[7];
        perRecHitData[6] = 1;
        continue;
      }

      bool matched = false;

      for (CSCRecHit2DCollection::const_iterator iHit = recHits->begin(); iHit != recHits->end(); ++iHit) {
        // for now, this throws with DT TrackingRecHits
        try {
          if (myHit->sharesInput(&(*iHit), CSCRecHit2D::all)) {
            ++counts_["AllMatchedRecHits"];
            ++perEventData[3];
            ++perMuonData[1];
            perRecHitData[0] = 1;
            matched = true;
          }
          if (myHit->sharesInput(&(*iHit), CSCRecHit2D::some)) {
            ++counts_["SomeMatchedRecHits"];
            ++perEventData[4];
            ++perMuonData[2];
            perRecHitData[1] = 1;
            matched = true;
          }
          if (myHit->sharesInput(&(*iHit), CSCRecHit2D::allWires)) {
            ++counts_["AllWiresMatchedRecHits"];
            ++perEventData[5];
            ++perMuonData[3];
            perRecHitData[2] = 1;
            matched = true;
          }
          if (myHit->sharesInput(&(*iHit), CSCRecHit2D::someWires)) {
            ++counts_["SomeWiresMatchedRecHits"];
            ++perEventData[6];
            ++perMuonData[4];
            perRecHitData[3] = 1;
            matched = true;
          }
          if (myHit->sharesInput(&(*iHit), CSCRecHit2D::allStrips)) {
            ++counts_["AllStripsMatchedRecHits"];
            ++perEventData[7];
            ++perMuonData[5];
            perRecHitData[4] = 1;
            matched = true;
          }
          if (myHit->sharesInput(&(*iHit), CSCRecHit2D::someStrips)) {
            ++counts_["SomeStripsMatchedRecHits"];
            ++perEventData[8];
            ++perMuonData[6];
            perRecHitData[5] = 1;
            matched = true;
          }

        } catch (cms::Exception &e) {
          // Not a CSC RecHit, so I should break out of the loop to save time.
          break;
        }
      }

      if (!matched) {
        ++counts_["NotMatchedRecHits"];
        ++perEventData[9];
        ++perMuonData[7];
        perRecHitData[6] = 1;
      }

      ntuples_["perRecHit"]->Fill(perRecHitData);
    }

    ntuples_["perMuon"]->Fill(perMuonData);
  }

  ntuples_["perEvent"]->Fill(perEventData);
}

void CSCSharesInputTest::endJob() {
  std::cout << std::endl << "End of job statistics" << std::endl;
  for (std::map<std::string, uint64_t>::iterator iCount = counts_.begin(); iCount != counts_.end(); iCount++) {
    std::cout << iCount->first << ": " << iCount->second << std::endl;
  }
  std::cout << std::endl;
}

CSCSharesInputTest::~CSCSharesInputTest() {}

DEFINE_FWK_MODULE(CSCSharesInputTest);

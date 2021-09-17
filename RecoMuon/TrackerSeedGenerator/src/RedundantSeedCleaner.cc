/** \class RedundantSeedCleaner 
 * Description:
 * RedundantSeedCleaner (TrackerSeedGenerator) duplicate removal from triplets pairs pixel seeds .
 *
 * \author Alessandro Grelli, Jean-Roch Vlimant
*/

#include "RecoMuon/TrackerSeedGenerator/interface/RedundantSeedCleaner.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"

using namespace edm;
using namespace reco;

//
// definition of vectors
//

void RedundantSeedCleaner::define(std::vector<TrajectorySeed>& coll) {
  std::vector<TrajectorySeed> triplets;

  //search for triplest
  for (TrajectorySeedCollection::iterator itr = coll.begin(); itr != coll.end(); ++itr) {
    //fill vector of triplets
    if (itr->nHits() == 3)
      triplets.push_back(*itr);
  }

  // clean from shared input
  if (!triplets.empty())
    clean(triplets, coll);
}

//
// the sharedHits cleaner
//

void RedundantSeedCleaner::clean(const std::vector<TrajectorySeed>& seedTr, std::vector<TrajectorySeed>& seed) {
  // loop over triplets
  std::vector<TrajectorySeed> result;

  std::vector<bool> maskPairs = std::vector<bool>(seed.size(), true);
  int iPair = 0;

  for (TrajectorySeedCollection::iterator s1 = seed.begin(); s1 != seed.end(); ++s1) {
    //rechits from seed

    for (TrajectorySeedCollection::const_iterator s2 = seedTr.begin(); s2 != seedTr.end(); ++s2) {
      //empty
      if (s2->nHits() == 0)
        continue;

      //number of shared hits;
      int shared = 0;

      for (auto const& h2 : s2->recHits()) {
        for (auto const& h1 : s1->recHits()) {
          if (h2.sharesInput(&h1, TrackingRecHit::all))
            shared++;
          if (s1->nHits() != 3)
            LogDebug(theCategory) << shared << " shared hits counter if 2 erease the seed.";
        }
      }

      if (shared == 2 && s1->nHits() != 3) {
        maskPairs[iPair] = false;
      }

    }  //end triplets loop
    ++iPair;
  }  // end pairs loop

  iPair = 0;
  //remove pairs in triplets
  for (TrajectorySeedCollection::iterator s1 = seed.begin(); s1 != seed.end(); ++s1) {
    if (maskPairs[iPair])
      result.push_back(*s1);
    ++iPair;
  }

  // cleaned collection
  seed.swap(result);
}

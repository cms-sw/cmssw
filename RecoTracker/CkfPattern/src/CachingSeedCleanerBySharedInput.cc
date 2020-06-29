#include "RecoTracker/CkfPattern/interface/CachingSeedCleanerBySharedInput.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include <algorithm>

void CachingSeedCleanerBySharedInput::init(const std::vector<Trajectory> *vect) {
  theVault.clear();
  theCache.clear();
}

void CachingSeedCleanerBySharedInput::done() {
  //edm::LogInfo("CachingSeedCleanerBySharedInput") << " Calls: " << calls_ << ", Tracks: " << tracks_ <<", Comps: " << comps_ << " Vault: " << theVault.size() << ".";
  //calls_ = comps_ = tracks_ = 0;
  theVault.clear();
  theCache.clear();
}

void CachingSeedCleanerBySharedInput::add(const Trajectory *trj) {
  unsigned int idx = theVault.size();
  theVault.resize(idx + 1);
  // a vector of shared pointers....
  auto &hits = theVault.back();
  (*trj).validRecHits(hits);

  for (auto const &h : hits) {
    auto detid = h->geographicalId().rawId();

    //For seeds that are made only of pixel hits, it is pointless to store the
    //information about hits on other sub-detector of the trajectory.
    if (theOnlyPixelHits && h->geographicalId().subdetId() != PixelSubdetector::PixelBarrel &&
        h->geographicalId().subdetId() != PixelSubdetector::PixelEndcap)
      continue;
    if (detid)
      theCache.insert(std::pair<uint32_t, unsigned int>(detid, idx));
  }
}

bool CachingSeedCleanerBySharedInput::good(const TrajectorySeed *seed) {
  if (seed->nHits() == 0) {
    return true;
  }

  auto const &range = seed->recHits();

  auto first = range.begin();
  auto last = range.end();
  auto detid = first->geographicalId().rawId();

  //calls_++;
  auto itrange = theCache.equal_range(detid);
  for (auto it = itrange.first; it != itrange.second; ++it) {
    assert(it->first == detid);
    //tracks_++;

    // seeds are limited to the first "theNumHitsForSeedCleaner" hits in trajectory...
    int ext = std::min(theNumHitsForSeedCleaner, int(theVault[it->second].size()));
    auto ts = theVault[it->second].begin();
    auto te = ts + ext;
    auto t = ts;
    auto curr = first;
    for (; curr != last; ++curr) {
      bool found = false;
      for (; t != te; ++t) {
        //comps_++;
        if (curr->sharesInput((**t).hit(), TrackingRecHit::all)) {
          found = true;
          ++t;
          break;
        }
      }
      if (!found)
        break;
    }
    if (curr == last)
      return false;
  }
  return true;
}

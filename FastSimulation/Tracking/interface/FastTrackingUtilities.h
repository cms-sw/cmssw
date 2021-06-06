
#ifndef FASTSIMULATION_TRACKING_FASTTRACKINGHELPER_H
#define FASTSIMULATION_TRACKING_FASTTRACKINGHELPER_H

#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"

namespace fastTrackingUtilities {

  template <class T>
  inline void setRecHitCombinationIndex(edm::OwnVector<T> &recHits, int32_t icomb) {
    for (auto &recHit : recHits) {
      if (!trackerHitRTTI::isFast(recHit)) {
        throw cms::Exception("fastTrackingHelpers::setRecHitCombinationIndex: one of hits in OwnVector is non-fastsim");
      }
      static_cast<FastTrackerRecHit &>(recHit).setRecHitCombinationIndex(icomb);
    }
  }

  // get recHitCombination for TrackCandidate and TrajectorySeed
  template <class T>
  int32_t getRecHitCombinationIndex(const T &object) {
    // seed must have at least one hit
    if (object.recHits().empty()) {
      throw cms::Exception("fastTrackingHelpers::getRecHitCombinationIndex")
          << "  given object has 0 hits" << std::endl;
    }

    const TrackingRecHit &recHit = *object.recHits().begin();
    if (!trackerHitRTTI::isFast(recHit)) {
      throw cms::Exception("fastTrackingHelpers::setRecHitCombinationIndex")
          << "  one of hits in OwnVector is non-fastsim" << std::endl;
    }
    // cast and return combination index
    return static_cast<const FastTrackerRecHit &>(recHit).recHitCombinationIndex();
    // return combination index of first hit
  }

  inline bool hitIsMasked(const FastTrackerRecHit *hit, const std::vector<bool> &hitMasks) {
    for (unsigned int i = 0; i < hit->nIds(); i++) {
      if (!(hitMasks)[hit->id(i)]) {
        return false;
      }
    }
    return true;
  }

  inline double hitLocalError(const TrackingRecHit *hit) {
    double xx = hit->localPositionError().xx();
    double yy = hit->localPositionError().yy();
    double xy = hit->localPositionError().xy();
    double delta = std::sqrt((xx - yy) * (xx - yy) + 4. * xy * xy);
    return 0.5 * (xx + yy - delta);
  }
}  // namespace fastTrackingUtilities

#endif

#ifndef SEEDMATCHER_H
#define SEEDMATCHER_H

#include "vector"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHitCollection.h"

class TrajectorySeed;
class Propagator;
class MagneticField;
class TrajectoryStateOnSurface;
class SimTrack;
class TrackerGeometry;

class SeedMatcher {
public:
  static std::vector<int> matchRecHitCombinations(
      const TrajectorySeed& seed,
      const FastTrackerRecHitCombinationCollection& recHitCombinationCollection,
      const std::vector<SimTrack>& simTrackCollection,
      double maxMatchEstimator,
      const Propagator& propagator,
      const MagneticField& magneticField,
      const TrackerGeometry& trackerGeometry);

  static double matchSimTrack(const TrajectoryStateOnSurface& seedState,
                              const SimTrack& simTrack,
                              const Propagator& propagator,
                              const MagneticField& magneticField);
};

#endif

#ifndef PixelTrackFitting_TracksWithHits_H
#define PixelTrackFitting_TracksWithHits_H

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

#include <vector>

namespace pixeltrackfitting {
  typedef std::pair<reco::Track*, SeedingHitSet> TrackWithRecHits;
  typedef std::vector<TrackWithRecHits> TracksWithRecHits; 
}

#endif

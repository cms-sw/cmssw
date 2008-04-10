#ifndef PixelTrackFitting_TracksWithHits_H
#define PixelTrackFitting_TracksWithHits_H

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include <vector>

namespace pixeltrackfitting {
  typedef std::pair<reco::Track*, std::vector<const TrackingRecHit *> > TrackWithRecHits;
  typedef std::vector<TrackWithRecHits> TracksWithRecHits; 
}

#endif

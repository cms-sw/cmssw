#ifndef PixelTrackFitting_TracksWithHits_H
#define PixelTrackFitting_TracksWithHits_H

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include <vector>

namespace pixeltrackfitting {
  typedef std::pair<reco::Track*, std::vector<const TrackingRecHit *> > TrackWithRecHits;
  typedef std::vector<TrackWithRecHits> TracksWithRecHits; 
}

#endif

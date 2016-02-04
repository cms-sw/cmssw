#ifndef PixelFitter_H
#define PixelFitter_H

#include <vector>

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"


class PixelFitter {
public:
  virtual ~PixelFitter(){}
  virtual reco::Track* run(
      const edm::EventSetup& es,
      const std::vector<const TrackingRecHit *>& hits, 
      const TrackingRegion& region) const = 0;
};
#endif

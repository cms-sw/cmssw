#ifndef PixelFitterByConformalMappingAndLine_H
#define PixelFitterByConformalMappingAndLine_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"

class PixelFitterByConformalMappingAndLine : public PixelFitter {
public:
  PixelFitterByConformalMappingAndLine();
  virtual ~PixelFitterByConformalMappingAndLine() { }
  virtual reco::Track* run( 
      const edm::EventSetup& es,
      const std::vector<const TrackingRecHit *>& hits, 
      const TrackingRegion& region) const;
};
#endif

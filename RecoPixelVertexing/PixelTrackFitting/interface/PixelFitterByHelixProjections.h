#ifndef PixelFitterByHelixProjections_H
#define PixelFitterByHelixProjections_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <vector>

class TransientTrackingRecHitBuilder;
class TrackerGeometry;
class MagneticField;

class PixelFitterByHelixProjections : public PixelFitter {
public:
  PixelFitterByHelixProjections();
  virtual ~PixelFitterByHelixProjections() {}
    virtual reco::Track* run(
      const edm::EventSetup& es,
      const std::vector<const TrackingRecHit *>& hits,
      const TrackingRegion& region) const;
private:
  int charge(const vector<GlobalPoint> & points) const;
  float cotTheta(const GlobalPoint& pinner, const GlobalPoint& pouter) const;
  float phi(float xC, float yC, int charge) const;
  float pt(float curvature) const;
  float zip(float d0, float curv, 
    const GlobalPoint& pinner, const GlobalPoint& pouter) const;
  double errZip2(float apt, float eta) const;
  double errTip2(float apt, float eta) const;

private:
  mutable const TrackerGeometry * theTracker;
  mutable const MagneticField * theField;
  mutable const TransientTrackingRecHitBuilder * theTTRecHitBuilder;

};
#endif

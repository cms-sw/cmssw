#ifndef TrackFitter_H
#define TrackFitter_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

class TransientTrackingRecHitBuilder;
class TrackerGeometry;
class MagneticField;

class TrackFitter : public PixelFitter
{
public:
  TrackFitter(const edm::ParameterSet& cfg);
  virtual ~TrackFitter() { }
  virtual reco::Track* run
    (const edm::EventSetup& es,
     const std::vector<const TrackingRecHit *>& hits,
     const TrackingRegion& region) const;

private:
  float getCotThetaAndUpdateZip
    (const GlobalPoint& inner, const GlobalPoint& outer,
     float radius, float phi, float d0, float& zip) const;
  float getPhi(float xC, float yC, int charge) const;
  float getZip(float d0, float curv, 
    const GlobalPoint& inner, const GlobalPoint& outer) const;
  void getErrTipAndErrZip(float pt, float eta,
                          float & errZip, float & errTip) const;

  edm::ParameterSet theConfig;

  mutable const TrackerGeometry * theTracker;
  mutable const MagneticField * theField;
  mutable const TransientTrackingRecHitBuilder * theTTRecHitBuilder;

};
#endif

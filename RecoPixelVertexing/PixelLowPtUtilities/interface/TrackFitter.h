#ifndef TrackFitter_H
#define TrackFitter_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterBase.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

class TransientTrackingRecHitBuilder;
class TrackerGeometry;
class MagneticField;

class TrackFitter : public PixelFitterBase
{
public:
  TrackFitter(const edm::EventSetup *es, const TrackerGeometry *tracker,
              const MagneticField *field, const TransientTrackingRecHitBuilder *ttrhBuilder):
    theES(es), theTracker(tracker), theField(field), theTTRecHitBuilder(ttrhBuilder)
  {}
  virtual ~TrackFitter() { }

  std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit *>& hits, const TrackingRegion& region) const override;

private:
  float getCotThetaAndUpdateZip
    (const GlobalPoint& inner, const GlobalPoint& outer,
     float radius, float phi, float d0, float& zip) const;
  float getPhi(float xC, float yC, int charge) const;
  float getZip(float d0, float curv, 
    const GlobalPoint& inner, const GlobalPoint& outer) const;
  void getErrTipAndErrZip(float pt, float eta,
                          float & errZip, float & errTip) const;

  const edm::EventSetup *theES;
  const TrackerGeometry * theTracker;
  const MagneticField * theField;
  const TransientTrackingRecHitBuilder * theTTRecHitBuilder;

};
#endif

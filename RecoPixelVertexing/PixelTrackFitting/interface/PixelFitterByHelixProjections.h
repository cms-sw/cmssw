#ifndef PixelFitterByHelixProjections_H
#define PixelFitterByHelixProjections_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterBase.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

class TrackerTopology;

class PixelFitterByHelixProjections final : public PixelFitterBase {
public:
  explicit PixelFitterByHelixProjections(const edm::EventSetup *es, const MagneticField *field,
                                         bool scaleErrorsForBPix1, float scaleFactor);
  virtual ~PixelFitterByHelixProjections() {}
  virtual std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit *>& hits,
                                           const TrackingRegion& region) const override;

private:
  const edm::EventSetup *theES;
  const MagneticField *theField;
  const bool thescaleErrorsForBPix1;
  const float thescaleFactor;
  TrackerTopology const * theTopo=nullptr;
};
#endif

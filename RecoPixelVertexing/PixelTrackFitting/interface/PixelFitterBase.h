#ifndef RecoPixelVertexing_PixelTrackFitting_PixelFitterBase_H
#define RecoPixelVertexing_PixelTrackFitting_PixelFitterBase_H

#include "DataFormats/TrackReco/interface/Track.h"

#include <vector>
#include <memory>

namespace edm {class ParameterSet; class Event; class EventSetup;}
class TrackingRegion;
class TrackingRecHit;


class PixelFitterBase {
public:
  virtual ~PixelFitterBase(){}

  virtual std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit *>& hits,
                                           const TrackingRegion& region) const { return std::unique_ptr<reco::Track>(); }

  virtual reco::Track* run(
      const edm::EventSetup& es,
      const std::vector<const TrackingRecHit *>& hits,
      const TrackingRegion& region) const { return 0;}

  virtual reco::Track* run(
      const edm::Event& ev,
      const edm::EventSetup& es,
      const std::vector<const TrackingRecHit *>& hits,
      const TrackingRegion& region) const { return run(es,hits,region); }
};
#endif

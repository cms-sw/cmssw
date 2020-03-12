#ifndef RecoPixelVertexing_PixelTrackFitting_PixelFitterBase_H
#define RecoPixelVertexing_PixelTrackFitting_PixelFitterBase_H

#include "DataFormats/TrackReco/interface/Track.h"

#include <vector>
#include <memory>

namespace edm {
  class EventSetup;
}
class TrackingRegion;
class TrackingRecHit;

class PixelFitterBase {
public:
  virtual ~PixelFitterBase() {}

  virtual std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit*>& hits,
                                           const TrackingRegion& region,
                                           const edm::EventSetup& setup) const = 0;
};
#endif

#ifndef RecoPixelVertexing_PixelTrackFitting_PixelFitterBase_H
#define RecoPixelVertexing_PixelTrackFitting_PixelFitterBase_H

#include <vector>

namespace edm {class ParameterSet; class Event; class EventSetup;}
namespace reco { class Track;}
class TrackingRegion;
class TrackingRecHit;



class PixelFitterBase {
public:
  virtual ~PixelFitterBase(){}

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

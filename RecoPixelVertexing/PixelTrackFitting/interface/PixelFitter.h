#ifndef PixelFitter_H
#define PixelFitter_H

#include <vector>

namespace edm {class ParameterSet; class Event; class EventSetup;}
namespace reco { class Track;}
class TrackingRegion;
class TrackingRecHit;



class PixelFitter {
public:
  virtual ~PixelFitter(){}

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

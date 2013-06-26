#ifndef PixelTrackFitting_PixelTrackFilter_H
#define PixelTrackFitting_PixelTrackFilter_H

namespace reco { class Track; }
namespace edm { class Event; }
class TrackingRecHit;

#include <vector>


class PixelTrackFilter {
public:
  virtual ~PixelTrackFilter() {}
  typedef std::vector<const TrackingRecHit *> Hits;
  virtual void update(edm::Event& ev) {}
  virtual bool operator()(const reco::Track*) const {return false;}
  virtual bool operator()(const reco::Track*, const Hits&) const {return false;} 
};
#endif

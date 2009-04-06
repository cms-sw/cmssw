#ifndef PixelTrackFitting_PixelTrackFilter_H
#define PixelTrackFitting_PixelTrackFilter_H

namespace reco { class Track; }
class TrackingRecHit;

#include <vector>


class PixelTrackFilter {
public:
  virtual ~PixelTrackFilter() {}
  typedef std::vector<const TrackingRecHit *> Hits;
  virtual bool operator()(const reco::Track*) const {return false;}
  virtual bool operator()(const reco::Track*, const Hits&) const {return false;} 
};
#endif

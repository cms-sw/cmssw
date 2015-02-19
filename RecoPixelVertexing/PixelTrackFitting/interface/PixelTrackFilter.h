#ifndef PixelTrackFitting_PixelTrackFilter_H
#define PixelTrackFitting_PixelTrackFilter_H

namespace reco { class Track; }
namespace edm { class Event; class EventSetup; class ConsumesCollector;}
class TrackingRecHit;

#include <vector>
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

class PixelTrackFilter {
public:
  virtual ~PixelTrackFilter() {}
  typedef std::vector<const TrackingRecHit *> Hits;
  virtual void update(const edm::Event& ev, const edm::EventSetup& es) = 0;
  virtual bool operator()(const reco::Track*) const {return false;}
  virtual bool operator()(const reco::Track*, const Hits&) const {return false;} 
  virtual bool operator()(const reco::Track*, const Hits&, const TrackerTopology *tTopo) const {return false;} 
};
#endif

#ifndef _TripletFilter_h_
#define _TripletFilter_h_

#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include <vector>

namespace edm { class EventSetup; }
class TrackingRecHit;
class ClusterShapeHitFilter;
class TrackerTopology;
class SiPixelClusterShapeCache;

class TripletFilter 
{
 public:
  TripletFilter(const edm::EventSetup& es);
  ~TripletFilter();
  bool checkTrack(const std::vector<const TrackingRecHit*>& recHits,
                  const std::vector<LocalVector>& localDirs,const TrackerTopology *tTopo, const SiPixelClusterShapeCache& clusterShapeCache);
  bool checkTrack(const std::vector<const TrackingRecHit*>& recHits,
                  const std::vector<GlobalVector>& globalDirs, const TrackerTopology *tTopo, const SiPixelClusterShapeCache& clusterShapeCache);

 private:
  const ClusterShapeHitFilter * theFilter;
};

#endif


#ifndef _TripletFilter_h_
#define _TripletFilter_h_

#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include <vector>

namespace edm { class EventSetup; }
class TrackingRecHit;
class ClusterShapeHitFilter;

class TripletFilter 
{
 public:
  TripletFilter(const edm::EventSetup& es);
  ~TripletFilter();
  bool checkTrack(std::vector<const TrackingRecHit*> recHits,
                  std::vector<LocalVector> localDirs);
  bool checkTrack(std::vector<const TrackingRecHit*> recHits,
                  std::vector<GlobalVector> globalDirs);

 private:
  const ClusterShapeHitFilter * theFilter;
};

#endif


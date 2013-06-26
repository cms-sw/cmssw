#ifndef _ClusterShapeTrackFilter_h_
#define _ClusterShapeTrackFilter_h_


#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"

#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/Vector2DBase.h"
typedef Vector2DBase<float,GlobalTag> Global2DVector;

//#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include <vector>

namespace edm { class ParameterSet; class EventSetup; }

class TrackerGeometry;
class TrackingRecHit;
class ClusterShapeHitFilter;
class TrackerTopology;

class ClusterShapeTrackFilter : public PixelTrackFilter 
{
 public:
  ClusterShapeTrackFilter(const edm::ParameterSet& ps,
                          const edm::EventSetup& es);
  virtual ~ClusterShapeTrackFilter();
  virtual bool operator()
    (const reco::Track*, const std::vector<const TrackingRecHit *> &hits, 
     const TrackerTopology *tTopo) const;

 private:
  float areaParallelogram(const Global2DVector & a,
                          const Global2DVector & b) const;
  std::vector<GlobalVector>
    getGlobalDirs(const std::vector<GlobalPoint> & globalPoss) const;
  std::vector<GlobalPoint>
    getGlobalPoss(const std::vector<const TrackingRecHit *>& recHits) const;
 
  const TrackerGeometry * theTracker;
  const ClusterShapeHitFilter * theFilter;

  double ptMin;
  double ptMax;
};

#endif


#ifndef _ClusterShapeTrackFilter_h_
#define _ClusterShapeTrackFilter_h_


#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"

#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/Vector2DBase.h"
typedef Vector2DBase<float,GlobalTag> Global2DVector;

//#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

#include <vector>

namespace edm { class ParameterSet; class EventSetup; class Event;}

class TrackerGeometry;
class TrackingRecHit;
class ClusterShapeHitFilter;
class TrackerTopology;
class SiPixelClusterShapeCache;

class ClusterShapeTrackFilter : public PixelTrackFilter 
{
 public:
  ClusterShapeTrackFilter(const edm::ParameterSet& ps,
                          edm::ConsumesCollector& iC);
  virtual ~ClusterShapeTrackFilter();
  void update(const edm::Event& ev, const edm::EventSetup& es) override;
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

  edm::EDGetTokenT<SiPixelClusterShapeCache> theClusterShapeCacheToken;
 
  const TrackerGeometry * theTracker;
  const ClusterShapeHitFilter * theFilter;
  const SiPixelClusterShapeCache *theClusterShapeCache;

  double ptMin;
  double ptMax;
};

#endif


#ifndef _LowPtClusterShapeSeedComparitor_h_
#define _LowPtClusterShapeSeedComparitor_h_


#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

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
//class ClusterShapeHitFilter;

class LowPtClusterShapeSeedComparitor : public SeedComparitor
{
 public:
  LowPtClusterShapeSeedComparitor(const edm::ParameterSet& ps);
  virtual ~LowPtClusterShapeSeedComparitor();
  virtual void init(const edm::EventSetup& es) ;
  virtual bool compatible(const SeedingHitSet  &hits, const TrackingRegion & region);
  //not sure if we need methods below or if they are for VI changes
  virtual bool compatible(const TrajectorySeed &seed) const { return true; }
  virtual bool compatible(const TrajectoryStateOnSurface &,
                          const TransientTrackingRecHit::ConstRecHitPointer &hit) const { return true; }
  virtual bool compatible(const SeedingHitSet  &hits,
                          const GlobalTrajectoryParameters &helixStateAtVertex,
                          const FastHelix                  &helix,
                          const TrackingRegion & region) const { return true; }
  virtual bool compatible(const SeedingHitSet  &hits,
                          const GlobalTrajectoryParameters &straightLineStateAtVertex,
                          const TrackingRegion & region) const { return true; }

 private:
  float areaParallelogram(const Global2DVector & a,
                          const Global2DVector & b);
  std::vector<GlobalVector>
    getGlobalDirs(const std::vector<GlobalPoint> & globalPoss);
  std::vector<GlobalPoint>
    getGlobalPoss(const TransientTrackingRecHit::ConstRecHitContainer & recHits);
 
   //const ClusterShapeHitFilter * theFilter;
   edm::ESHandle<ClusterShapeHitFilter> theShapeFilter;
};

#endif


#include "RecoPixelVertexing/PixelLowPtUtilities/interface/LowPtClusterShapeSeedComparitor.h"

#include "RecoPixelVertexing/PixelTrackFitting/src/CircleFromThreePoints.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/HitInfo.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

inline float sqr(float x) { return x*x; }

using namespace std;

/*****************************************************************************/
LowPtClusterShapeSeedComparitor::LowPtClusterShapeSeedComparitor
  (const edm::ParameterSet& ps)
{

}

/*****************************************************************************/
LowPtClusterShapeSeedComparitor::~LowPtClusterShapeSeedComparitor()
{
}

/*****************************************************************************/
float LowPtClusterShapeSeedComparitor::areaParallelogram
  (const Global2DVector& a, const Global2DVector& b)
{  
  return a.x() * b.y() - a.y() * b.x();
}

/*****************************************************************************/
vector<GlobalVector> LowPtClusterShapeSeedComparitor::getGlobalDirs
  (const vector<GlobalPoint> & g)
{
  // Get 2d points
  vector<Global2DVector> p;
  for(vector<GlobalPoint>::const_iterator ig = g.begin();
                                          ig!= g.end(); ig++)
     p.push_back( Global2DVector(ig->x(), ig->y()) );

  //
  vector<GlobalVector> globalDirs;

  // Determine circle
  CircleFromThreePoints circle(g[0],g[1],g[2]);

  if(circle.curvature() != 0.)
  {
    Global2DVector c (circle.center().x(), circle.center().y());

    float rad2 = (p[0] - c).mag2();
    float a12 = asin(fabsf(areaParallelogram(p[0] - c, p[1] - c)) / rad2);

    float slope = (g[1].z() - g[0].z()) / a12;

    float cotTheta = slope * circle.curvature(); // == sinhEta
    float coshEta  = sqrt(1 + sqr(cotTheta));    // == 1/sinTheta

    // Calculate globalDirs
    float sinTheta =       1. / coshEta;
    float cosTheta = cotTheta * sinTheta;

    int dir;
    if(areaParallelogram(p[0] - c, p[1] - c) > 0) dir = 1; else dir = -1;

    float curvature = circle.curvature();

    for(vector<Global2DVector>::const_iterator ip = p.begin();
                                               ip!= p.end(); ip++)
    {
      Global2DVector v = (*ip - c)*curvature*dir;
      globalDirs.push_back(GlobalVector(-v.y()*sinTheta,
                                         v.x()*sinTheta,
                                               cosTheta));
    }
  }

  return globalDirs;
}

/*****************************************************************************/
vector<GlobalPoint> LowPtClusterShapeSeedComparitor::getGlobalPoss
(const TransientTrackingRecHit::ConstRecHitContainer & thits)
{
  vector<GlobalPoint> globalPoss;

  for(TransientTrackingRecHit::ConstRecHitContainer::const_iterator recHit = thits.begin();
      recHit!= thits.end();
      recHit++)
  {
    DetId detId = (*recHit)->hit()->geographicalId();

    GlobalPoint gpos = (*recHit)->globalPosition();

    globalPoss.push_back(gpos);
  }

  return globalPoss;
}

/*****************************************************************************/
bool LowPtClusterShapeSeedComparitor::compatible(const SeedingHitSet &hits,
					    const edm::EventSetup &es)
//(const reco::Track* track, const vector<const TrackingRecHit *> & recHits) const
{

  // Get cluster shape hit filter
  edm::ESHandle<ClusterShapeHitFilter> shape;
  es.get<CkfComponentsRecord>().get("ClusterShapeHitFilter",shape);
  theFilter = shape.product();

  const TransientTrackingRecHit::ConstRecHitContainer & thits = hits.container();

  // Get global positions
  vector<GlobalPoint>  globalPoss = getGlobalPoss(thits);

  // Get global directions
  vector<GlobalVector> globalDirs = getGlobalDirs(globalPoss);

  bool ok = true;

  // Check whether shape of pixel cluster is compatible
  // with local track direction
  for(int i = 0; i < 3; i++)
  {
    const SiPixelRecHit* pixelRecHit =
      dynamic_cast<const SiPixelRecHit *>(thits[i]->hit());

    if(!pixelRecHit->isValid())
    { 
      ok = false; break; 
    }

    if(! theFilter->isCompatible(*pixelRecHit, globalDirs[i]) )
    {
      LogTrace("MinBiasTracking")
         << "  [LowPtClusterShapeSeedComparitor] clusShape problem"
         << HitInfo::getInfo(*thits[i]->hit());

      ok = false; break;
    }
  }

  return ok;
}


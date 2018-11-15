#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrackFilter.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/HitInfo.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/CircleFromThreePoints.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

inline float sqr(float x) { return x*x; }

using namespace std;

/*****************************************************************************/
ClusterShapeTrackFilter::ClusterShapeTrackFilter(const SiPixelClusterShapeCache *cache, double ptmin, double ptmax, const edm::EventSetup& es):
  theClusterShapeCache(cache),
  ptMin(ptmin), ptMax(ptmax)
{
  // Get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  theTracker = tracker.product();

  // Get cluster shape hit filter
  edm::ESHandle<ClusterShapeHitFilter> shape;
  es.get<CkfComponentsRecord>().get("ClusterShapeHitFilter",shape);
  theFilter = shape.product();

  edm::ESHandle<TrackerTopology> tTopoHand;
  es.get<TrackerTopologyRcd>().get(tTopoHand);
  tTopo = tTopoHand.product();
}

/*****************************************************************************/
ClusterShapeTrackFilter::~ClusterShapeTrackFilter()
{
}

/*****************************************************************************/
float ClusterShapeTrackFilter::areaParallelogram
  (const Global2DVector& a, const Global2DVector& b) const
{  
  return a.x() * b.y() - a.y() * b.x();
}

/*****************************************************************************/
vector<GlobalVector> ClusterShapeTrackFilter::getGlobalDirs
  (const vector<GlobalPoint> & g) const
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
vector<GlobalPoint> ClusterShapeTrackFilter::getGlobalPoss
  (const vector<const TrackingRecHit *> & recHits) const
{
  vector<GlobalPoint> globalPoss;

  for(vector<const TrackingRecHit *>::const_iterator recHit = recHits.begin();
                                                     recHit!= recHits.end();
                                                     recHit++)
  {
    DetId detId = (*recHit)->geographicalId();

    GlobalPoint gpos = 
      theTracker->idToDet(detId)->toGlobal((*recHit)->localPosition());

    globalPoss.push_back(gpos);
  }

  return globalPoss;
}

/*****************************************************************************/
bool ClusterShapeTrackFilter::operator()
  (const reco::Track* track,
   const vector<const TrackingRecHit *> & recHits) const
{
  // Do not even look at pairs
  if(recHits.size() <= 2) return true;

  // Check pt
  if(track->pt() < ptMin ||
     track->pt() > ptMax)
  {
    LogTrace("ClusterShapeTrackFilter")
       << "  [ClusterShapeTrackFilter] pt not in range: "
       << ptMin << " " << track->pt() << " " << ptMax;
    return false;
  }

  // Get global positions
  vector<GlobalPoint>  globalPoss = getGlobalPoss(recHits);

  // Get global directions
  vector<GlobalVector> globalDirs = getGlobalDirs(globalPoss);
  if ( globalDirs.empty() ) return false;

  bool ok = true;

  // Check whether shape of pixel cluster is compatible
  // with local track direction
  for(unsigned int i = 0; i < recHits.size(); i++)
  {
    const SiPixelRecHit* pixelRecHit =
      dynamic_cast<const SiPixelRecHit *>(recHits[i]);

    if(!pixelRecHit->isValid())
    { 
      ok = false; break; 
    }

    if(! theFilter->isCompatible(*pixelRecHit, globalDirs[i], *theClusterShapeCache) )
    {
      LogTrace("ClusterShapeTrackFilter")
         << "  [ClusterShapeTrackFilter] clusShape problem"
         << HitInfo::getInfo(*recHits[i],tTopo);

      ok = false; break;
    }
  }

  return ok;
}


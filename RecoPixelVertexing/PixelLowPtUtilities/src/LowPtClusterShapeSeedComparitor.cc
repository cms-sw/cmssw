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
#include<cmath>

/*****************************************************************************/
LowPtClusterShapeSeedComparitor::LowPtClusterShapeSeedComparitor
  (const edm::ParameterSet&)
{
}

/*****************************************************************************/
LowPtClusterShapeSeedComparitor::~LowPtClusterShapeSeedComparitor()
{
}

namespace {
  inline float sqr(float x) { return x*x; }

  /*****************************************************************************/
  inline
  float areaParallelogram
  (const Global2DVector& a, const Global2DVector& b)
  {  
    return a.x() * b.y() - a.y() * b.x();
  }
  
  /*****************************************************************************/

  inline 
  bool getGlobalDirs(GlobalPoint const * g,GlobalVector * globalDirs)
  {
    
    
    // Determine circle
    CircleFromThreePoints circle(g[0],g[1],g[2]);
    
    float curvature = circle.curvature();
    if(curvature = 0.) {
      LogDebug("LowPtClusterShapeSeedComparitor")<<"the curvature is null:"
						 <<"\n point1: "<<g[0]
						 <<"\n point2: "<<g[1]
						 <<"\n point3: "<<g[2];
      return false;
    }

   // Get 2d points
    Global2DVector p[3];
    for(int i=0; i<3; i++)
      p[i] = Global2DVector(g[i].x(), g[i].y());
 
    Global2DVector c (circle.center().x(), circle.center().y());
    
    float rad2 = (p[0] - c).mag2();
    float area = std::abs(areaParallelogram(p[1] - p[0], p[1] - c));
    
    float a12;
    const float pi2 = M_PI/2;
    if(area >= rad2) a12 = pi2;
    else a12 = std::asin(area / rad2);
    
    float slope = (g[1].z() - g[0].z()) / a12;
    
    float cotTheta = slope * curvature; // == sinhEta
    float coshEta  = std::sqrt(1.f + sqr(cotTheta));    // == 1/sinTheta
    
    // Calculate globalDirs
    float sinTheta =       1. / coshEta;
    float cosTheta = cotTheta * sinTheta;
    
    flot dir = (areaParallelogram(p[0] - c, p[1] - c) > 0) ? 1 : -1;
        
    for(int ip = 0; ip!=3;  ip++) {
      Global2DVector v = (p[ip] - c)*(curvature*dir*sinTheta);
      globalDirs[ip] = GlobalVector(-v.y(),
				    v.x(),
				    cosTheta
				    );
    }
    return true;
  }

  /*****************************************************************************/

  
  inline
  void getGlobalPos(const SeedingHitSet &hits, GlobalPoint * globalPoss)
  {
    
    for(unsigned int i=0; i!=hits.size(); ++i)
  	globalPoss[i] = hits[i]->globalPosition();
  }

} // namespace

/*****************************************************************************/
bool LowPtClusterShapeSeedComparitor::compatible(const SeedingHitSet &hits,
					    const edm::EventSetup &es)
//(const reco::Track* track, const vector<const TrackingRecHit *> & recHits) const
{
  assert(hits.size()==3);

  // Get cluster shape hit filter
  edm::ESHandle<ClusterShapeHitFilter> shape;
  es.get<CkfComponentsRecord>().get("ClusterShapeHitFilter",shape);
   const ClusterShapeHitFilter * theFilter = shape.product();

   // Get global positions
   GlobalPoint  globalPoss[3];
   getGlobalPos(hits, globalPoss);

  // Get global directions
  GlobalVector globalDirs[3]; 

  bool ok = getGlobalDirs(globalPoss,globalDirs);

  // Check whether shape of pixel cluster is compatible
  // with local track direction

  if (!ok)
    {
      LogDebug("LowPtClusterShapeSeedComparitor")<<"curvarture 0:"
						 <<"\nnHits: "<<hits.size()
						 <<" will say the seed is good anyway.";
      return true;
    }

  for(int i = 0; i < 3; i++)
  {
    const SiPixelRecHit* pixelRecHit =
      dynamic_cast<const SiPixelRecHit *>(hits[i]->hit());

    if (!pixelRecHit){
      edm::LogError("LowPtClusterShapeSeedComparitor")<<"this is not a pixel cluster";
      ok = false; break;
    }

    if(!pixelRecHit->isValid())
    { 
      ok = false; break; 
    }
    
    LogDebug("LowPtClusterShapeSeedComparitor")<<"about to compute compatibility."
					       <<"hit ptr: "<<pixelRecHit
					       <<"global direction:"<< globalDirs[i];


    if(! theFilter->isCompatible(*pixelRecHit, globalDirs[i]) )
    {
      LogTrace("LowPtClusterShapeSeedComparitor")
         << " clusShape is not compatible"
         << HitInfo::getInfo(*hits[i]->hit());

      ok = false; break;
    }
  }

  return ok;
}


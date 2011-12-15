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



#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/Basic2DVector.h"

#include<cmath>



namespace {
  typedef Basic2DVector<float>   Vector2D;

  inline float sqr(float x) { return x*x; }

  /*****************************************************************************/
  inline
  float areaParallelogram
  (const Vector2D& a, const Vector2D& b)
  {  
    return a.x() * b.y() - a.y() * b.x();
  }
  
  /*****************************************************************************/

  inline 
  bool getGlobalDirs(GlobalPoint const * g, GlobalVector * globalDirs)
  {
    
    
    // Determine circle
    CircleFromThreePoints circle(g[0],g[1],g[2]);
    
    float curvature = circle.curvature();
    if(0.f == curvature) {
      LogDebug("LowPtClusterShapeSeedComparitor")<<"the curvature is null:"
						 <<"\n point1: "<<g[0]
						 <<"\n point2: "<<g[1]
						 <<"\n point3: "<<g[2];
      return false;
    }

   // Get 2d points
    Vector2D p[3], v[3];
    for(int i=0; i!=3; i++)
      p[i] =  g[i].basicVector().xy();
 
    Vector2D c  = circle.center();
    for(int ip = 0; ip!=3;  ip++)
      v[ip] = p[ip] - c;    

    float rad2 = v[0].mag2();
    float area = std::abs(areaParallelogram(p[1] - p[0], v[1]));
    
    float a12;
    const float pi2 = M_PI/2;
    if(area >= rad2) a12 = pi2;
    else a12 = std::asin(area / rad2);
    
    float slope = (g[1].z() - g[0].z()) / a12;
 
    // Calculate globalDirs
   
    float cotTheta = slope * curvature; 
    float sinTheta = 1.f/std::sqrt(1.f + sqr(cotTheta));
    float cosTheta = cotTheta*sinTheta;
    
    if (areaParallelogram(v[0], v[1] ) < 0)  sinTheta = - sinTheta;
        
    for(int i = 0; i!=3;  i++) {
      Vector2D vl = v[i]*(curvature*sinTheta);
      globalDirs[i] = GlobalVector(-vl.y(),
				    vl.x(),
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


#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrajectoryFilter.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterData.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShape.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <fstream>

using namespace std;

#define Nsigma 5

/*****************************************************************************/
ClusterShapeTrajectoryFilter::ClusterShapeTrajectoryFilter
 (const TrackingGeometry* theTracker_,
  const MagneticField* theMagneticField_)
  : theTracker(theTracker_), theMagneticField(theMagneticField_)
{
  // Load pixel
  {
  edm::FileInPath
    fileInPath("RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape.par");
  ifstream inFile(fileInPath.fullPath().c_str());

  while(inFile.eof() == false)
  {
    int part,dx,dy;

    inFile >> part;
    inFile >> dx;
    inFile >> dy;

    for(int b = 0; b<2 ; b++) // branch
    for(int d = 0; d<2 ; d++) // direction
    for(int k = 0; k<2 ; k++) // lower and upper
      inFile >> pixelLimits[part][dx][dy][b][d][k];

    double f;
    inFile >> f; // density
    inFile >> f;

    int d;
    inFile >> d; // points
  }

  inFile.close();

  LogTrace("MinBiasTracking")
    << "[TrajectFilter] pixel-cluster-shape filter loaded";
  }
 
  {
  // Load strip
  edm::FileInPath
    fileInPath("RecoPixelVertexing/PixelLowPtUtilities/data/stripShape.par");
  ifstream inFile(fileInPath.fullPath().c_str());

  while(inFile.eof() == false)
  {
    int dx; float f;

    inFile >> dx;
    inFile >> stripLimits[dx][0];
    inFile >> f;
    inFile >> stripLimits[dx][1];
    inFile >> f;
  }

  inFile.close();
 
  LogTrace("MinBiasTracking")
    << "[TrajectFilter] strip-cluster-width filter loaded";
  }
}

/*****************************************************************************/
bool ClusterShapeTrajectoryFilter::isInside
  (const float a[2][2], pair<float,float> movement) const
{
  if(a[0][0] == a[0][1] && a[1][0] == a[1][1])
    return true;

  if(movement.first  > a[0][0] && movement.first  < a[0][1] &&
     movement.second > a[1][0] && movement.second < a[1][1])
    return true;
  else
    return false;
}

/*****************************************************************************/
bool ClusterShapeTrajectoryFilter::processHit
  (const GlobalVector gdir, const SiPixelRecHit* recHit) const
{
  DetId id = recHit->geographicalId();

  const PixelGeomDetUnit* pixelDet =
    dynamic_cast<const PixelGeomDetUnit*> (theTracker->idToDet(id));

  LocalVector ldir = pixelDet->toLocal(gdir);

  ClusterData data;
  ClusterShape theClusterShape;
  theClusterShape.getExtra(*pixelDet, *recHit, data);
  
  int dx = data.size.first;
  int dy = data.size.second;

  if(data.isStraight && data.isComplete && dx <= MaxSize && abs(dy) <= MaxSize)
  {
    int part   = (data.isInBarrel ? 0 : 1);
    int orient = (data.isNormalOriented ? 1 : -1);

    pair<float,float> movement;
    movement.first  = ldir.x() / (fabs(ldir.z()) * data.tangent.first )*orient;
    movement.second = ldir.y() / (fabs(ldir.z()) * data.tangent.second)*orient;
    
    if(dy < 0)
    { dy = abs(dy); movement.second = - movement.second; }

    return (isInside(pixelLimits[part][dx][dy][0], movement) ||
            isInside(pixelLimits[part][dx][dy][1], movement));
  }
  else
  {
    // Shape is not straight or not complete or too wide
    return true;
  }
}

/*****************************************************************************/
bool ClusterShapeTrajectoryFilter::processHit
  (const GlobalVector gdir, const SiStripRecHit2D* recHit) const
{
  DetId id = recHit->geographicalId();

  const StripGeomDetUnit* stripDet =
    dynamic_cast<const StripGeomDetUnit*> (theTracker->idToDet(id));
  // !!!!! Problem in case of RadialStriptolopgy  !!!!!
  float tangent = stripDet->specificTopology().localPitch(LocalPoint(0,0,0))/
                  stripDet->surface().bounds().thickness();

  LocalVector ldir = stripDet->toLocal(gdir);
  float pred  = ldir.x() / (fabs(ldir.z()) * tangent);

  const SiStripCluster& cluster = *(recHit->cluster());
  int meas  = cluster.amplitudes().size(); 

  LocalVector lbfield = (stripDet->surface()).toLocal(theMagneticField->inTesla(stripDet->surface().position()));
  double theTanLorentzAnglePerTesla = 0.032;
//  double theTanLorentzAnglePerTesla =
//        theLorentzAngle->getLorentzAngle(id.rawId());

  float dir_x =  theTanLorentzAnglePerTesla * lbfield.y();
 
  float drift = dir_x / tangent;

  bool normal;
  if(stripDet->type().subDetector() == GeomDetEnumerators::TIB ||
     stripDet->type().subDetector() == GeomDetEnumerators::TOB)
  { // barrel
    float perp0 = stripDet->toGlobal( Local3DPoint(0.,0.,0.) ).perp();
    float perp1 = stripDet->toGlobal( Local3DPoint(0.,0.,1.) ).perp();
    normal = (perp1 > perp0);
  }
  else
  { // endcap
    float rot = stripDet->toGlobal( LocalVector (0.,0.,1.) ).z();
    float pos = stripDet->toGlobal( Local3DPoint(0.,0.,0.) ).z();
    normal = (rot * pos > 0);
  }

  // Correct for ExB
  pred += (normal ? 1 : -1) * drift;

  if(meas <= 24)
  {
    float m = stripLimits[meas][0];
    float s = stripLimits[meas][1];

    if( (pred >  m - Nsigma * s && pred <  m + Nsigma * s) ||
        (pred > -m - Nsigma * s && pred < -m + Nsigma * s) )
      return true;
    else
      return false;
  }
  else
  {
    // Too wide
    return true;
  }
}

/*****************************************************************************/
bool ClusterShapeTrajectoryFilter::toBeContinued
  (Trajectory& trajectory) const 
{
  vector<TrajectoryMeasurement> tms = trajectory.measurements();

  for(vector<TrajectoryMeasurement>::const_iterator
       tm = tms.begin(); tm!= tms.end(); tm++)
  {
    const TransientTrackingRecHit* ttRecHit = &(*((*tm).recHit()));

    if(ttRecHit->isValid())
    {
      const TrackingRecHit * tRecHit = ttRecHit->hit();

      TrajectoryStateOnSurface ts = (*tm).updatedState();
      GlobalVector gdir = ts.globalDirection();

      if(ttRecHit->det()->subDetector() == GeomDetEnumerators::PixelBarrel ||
         ttRecHit->det()->subDetector() == GeomDetEnumerators::PixelEndcap)
      { // pixel
        const SiPixelRecHit* recHit =
           dynamic_cast<const SiPixelRecHit *>(tRecHit);

        if(recHit != 0)
          if(processHit(gdir, recHit) == false)
            return false;
      }
      else
      { // strip
        if(dynamic_cast<const SiStripMatchedRecHit2D *>(tRecHit)  != 0)
        { // glued
          const SiStripMatchedRecHit2D* recHit =
            dynamic_cast<const SiStripMatchedRecHit2D *>(tRecHit);

          if(recHit != 0)
          { 
            if(processHit(gdir, recHit->monoHit()  ) == false)
              return false;

            if(processHit(gdir, recHit->stereoHit()) == false)
              return false;
          }
        }
        else
        { // single
          if(dynamic_cast<const SiStripRecHit2D *>(tRecHit) != 0)
          { // normal
            const SiStripRecHit2D* recHit =
              dynamic_cast<const SiStripRecHit2D *>(tRecHit);
  
            if(recHit != 0)
              if(processHit(gdir, recHit) == false)
                return false;
          }
          else
          { // projected
            const ProjectedSiStripRecHit2D* recHit =
              dynamic_cast<const ProjectedSiStripRecHit2D *>(tRecHit);
 
            if(recHit != 0)
              if(processHit(gdir, &(recHit->originalHit())) == false)
                return false;
          }
        }
      }
    }
  }

  return true;
}

/*****************************************************************************/
bool ClusterShapeTrajectoryFilter::toBeContinued
  (TempTrajectory& trajectory) const 
{
  TempTrajectory::DataContainer tms = trajectory.measurements();

  for(TempTrajectory::DataContainer::const_iterator
       tm = tms.rbegin(); tm!= tms.rend(); --tm)
  {
    const TransientTrackingRecHit* ttRecHit = &(*((*tm).recHit()));

    if(ttRecHit->isValid())
    {
      const TrackingRecHit * tRecHit = ttRecHit->hit();

      TrajectoryStateOnSurface ts = (*tm).updatedState();
      GlobalVector gdir = ts.globalDirection();

      if(ttRecHit->det()->subDetector() == GeomDetEnumerators::PixelBarrel ||
         ttRecHit->det()->subDetector() == GeomDetEnumerators::PixelEndcap)
      { // pixel
        const SiPixelRecHit* recHit =
           dynamic_cast<const SiPixelRecHit *>(tRecHit);

        if(recHit != 0)
          if(processHit(gdir, recHit) == false)
          {
            LogTrace("MinBiasTracking")
              << "  [TrajectFilter] fail pixel";
            return false;
          }
      }
      else
      { // strip
        if(dynamic_cast<const SiStripMatchedRecHit2D *>(tRecHit)  != 0)
        { // glued
          const SiStripMatchedRecHit2D* recHit =
            dynamic_cast<const SiStripMatchedRecHit2D *>(tRecHit);

          if(recHit != 0)
          { 
            if(processHit(gdir, recHit->monoHit()  ) == false)
            {
              LogTrace("MinBiasTracking")
               << "  [TrajectFilter] fail strip matched 1st";
              return false;
            }

            if(processHit(gdir, recHit->stereoHit()) == false)
            {
              LogTrace("MinBiasTracking")
                << "  [TrajectFilter] fail strip matched 2nd";
              return false;
            }
          }
        }
        else
        { // single
          if(dynamic_cast<const SiStripRecHit2D *>(tRecHit) != 0)
          { // normal
            const SiStripRecHit2D* recHit =
              dynamic_cast<const SiStripRecHit2D *>(tRecHit);
  
            if(recHit != 0)
              if(processHit(gdir, recHit) == false)
              {
                LogTrace("MinBiasTracking")
                  << "  [TrajectFilter] fail strip single";
                return false;
              }
          }
          else
          { // projected
            const ProjectedSiStripRecHit2D* recHit =
              dynamic_cast<const ProjectedSiStripRecHit2D *>(tRecHit);
 
            if(recHit != 0)
              if(processHit(gdir, &(recHit->originalHit())) == false)
              {
                LogTrace("MinBiasTracking")
                  << "  [TrajectFilter] fail strip projected";
                return false;
              }
          }
        }
      }
    }
  }

  return true;
}

/*****************************************************************************/
bool ClusterShapeTrajectoryFilter::qualityFilter
  (const Trajectory& trajectory) const
{
return true;
/*
  const FreeTrajectoryState * fts =
    trajectory.lastMeasurement().updatedState().freeTrajectoryState();

  float pt2 = fts->momentum().perp2();

  if(pt2 > 0) return true;
         else return false;
*/
}

/*****************************************************************************/
bool ClusterShapeTrajectoryFilter::qualityFilter
  (const TempTrajectory& trajectory) const
{
return true;
/*
  const FreeTrajectoryState * fts =
    trajectory.lastMeasurement().updatedState().freeTrajectoryState();

  float pt2 = fts->momentum().perp2();

  if(pt2 > 0) return true;
         else return false;
*/
}

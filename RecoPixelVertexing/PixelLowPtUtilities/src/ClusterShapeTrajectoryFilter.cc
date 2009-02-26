#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrajectoryFilter.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include <vector>
using namespace std;

/*****************************************************************************/
ClusterShapeTrajectoryFilter::ClusterShapeTrajectoryFilter
 (const GlobalTrackingGeometry* theTracker_,
  const MagneticField* theMagneticField_)
  : theTracker(theTracker_), 
    theMagneticField(theMagneticField_)
{
  theFilter = new ClusterShapeHitFilter(theTracker, theMagneticField);
}

/*****************************************************************************/
ClusterShapeTrajectoryFilter::~ClusterShapeTrajectoryFilter()
{
  delete theFilter;
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
          return theFilter->isCompatible(*recHit, gdir);
      }
      else
      { // strip
        if(dynamic_cast<const SiStripMatchedRecHit2D *>(tRecHit)  != 0)
        { // glued
          const SiStripMatchedRecHit2D* recHit =
            dynamic_cast<const SiStripMatchedRecHit2D *>(tRecHit);

          if(recHit != 0)
          { 
            return (theFilter->isCompatible(*(recHit->monoHit())  , gdir) &&
                    theFilter->isCompatible(*(recHit->stereoHit()), gdir));
          }
        }
        else
        { // single
          if(dynamic_cast<const SiStripRecHit2D *>(tRecHit) != 0)
          { // normal
            const SiStripRecHit2D* recHit =
              dynamic_cast<const SiStripRecHit2D *>(tRecHit);
  
            if(recHit != 0)
              return theFilter->isCompatible(*recHit, gdir);
          }
          else
          { // projected
            const ProjectedSiStripRecHit2D* recHit =
              dynamic_cast<const ProjectedSiStripRecHit2D *>(tRecHit);
 
            if(recHit != 0)
              return theFilter->isCompatible(recHit->originalHit(), gdir);
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
          if(! theFilter->isCompatible(*recHit, gdir))
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
            if(! theFilter->isCompatible(*(recHit->monoHit()  ), gdir))
            {
              LogTrace("MinBiasTracking")
               << "  [TrajectFilter] fail strip matched 1st";
              return false;
            }

            if(! theFilter->isCompatible(*(recHit->stereoHit()), gdir))
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
              if(! theFilter->isCompatible(*recHit, gdir))
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
              if(! theFilter->isCompatible(recHit->originalHit(), gdir))
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
}

/*****************************************************************************/
bool ClusterShapeTrajectoryFilter::qualityFilter
  (const TempTrajectory& trajectory) const
{
  return true;
}


// VI January 2012: needs to be	migrated to use	cluster	directly


#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrajectoryFilter.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include <vector>
using namespace std;

/*****************************************************************************/
ClusterShapeTrajectoryFilter::ClusterShapeTrajectoryFilter(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC):
  theCacheToken(iC.consumes<SiPixelClusterShapeCache>(iConfig.getParameter<edm::InputTag>("cacheSrc"))),
  theFilter(nullptr)
{}

ClusterShapeTrajectoryFilter::~ClusterShapeTrajectoryFilter()
{
}

void ClusterShapeTrajectoryFilter::setEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<ClusterShapeHitFilter> shape;
  iSetup.get<TrajectoryFilter::Record>().get("ClusterShapeHitFilter", shape);
  theFilter = shape.product();

  edm::Handle<SiPixelClusterShapeCache> cache;
  iEvent.getByToken(theCacheToken, cache);
  theCache = cache.product();
}

/*****************************************************************************/
bool ClusterShapeTrajectoryFilter::toBeContinued
  (Trajectory& trajectory) const 
{
  assert(theCache);
  vector<TrajectoryMeasurement> tms = trajectory.measurements();

  for(vector<TrajectoryMeasurement>::const_iterator
       tm = tms.begin(); tm!= tms.end(); tm++)
  {
    const TrackingRecHit* ttRecHit = &(*((*tm).recHit()));

    if(ttRecHit->isValid())
    {
      const TrackingRecHit * tRecHit = ttRecHit->hit();

      TrajectoryStateOnSurface ts = (*tm).updatedState();
      const GlobalVector gdir = ts.globalDirection();

      if(ttRecHit->det()->subDetector()==GeomDetEnumerators::SubDetector::PixelBarrel ||
	 ttRecHit->det()->subDetector()==GeomDetEnumerators::SubDetector::PixelEndcap ||
	 ttRecHit->det()->subDetector()==GeomDetEnumerators::SubDetector::P1PXB ||
	 ttRecHit->det()->subDetector()==GeomDetEnumerators::SubDetector::P1PXEC ||
	 ttRecHit->det()->subDetector()==GeomDetEnumerators::SubDetector::P2PXEC) 
      { // pixel
        const SiPixelRecHit* recHit =
           dynamic_cast<const SiPixelRecHit *>(tRecHit);

        if(recHit != 0)
          return theFilter->isCompatible(*recHit, gdir, *theCache);
      }
      else if(GeomDetEnumerators::isTrackerStrip(ttRecHit->det()->subDetector()))
      { // strip
        if(dynamic_cast<const SiStripMatchedRecHit2D *>(tRecHit)  != 0)
        { // glued
          const SiStripMatchedRecHit2D* recHit =
            dynamic_cast<const SiStripMatchedRecHit2D *>(tRecHit);

          if(recHit != 0)
          { 
            return (theFilter->isCompatible(recHit->monoHit()  , gdir) &&
                    theFilter->isCompatible(recHit->stereoHit(), gdir));
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
  assert(theCache);
  TempTrajectory::DataContainer tms = trajectory.measurements();

  for(TempTrajectory::DataContainer::const_iterator
       tm = tms.rbegin(); tm!= tms.rend(); --tm)
  {
    const TrackingRecHit* ttRecHit = &(*((*tm).recHit()));

    if(ttRecHit->isValid())
    {
      const TrackingRecHit * tRecHit = ttRecHit->hit();

      TrajectoryStateOnSurface ts = (*tm).updatedState();
      GlobalVector gdir = ts.globalDirection();

      if(ttRecHit->det()->subDetector()==GeomDetEnumerators::SubDetector::PixelBarrel ||
	 ttRecHit->det()->subDetector()==GeomDetEnumerators::SubDetector::PixelEndcap ||
	 ttRecHit->det()->subDetector()==GeomDetEnumerators::SubDetector::P1PXB ||
	 ttRecHit->det()->subDetector()==GeomDetEnumerators::SubDetector::P1PXEC ||
	 ttRecHit->det()->subDetector()==GeomDetEnumerators::SubDetector::P2PXEC) 
      { // pixel
        const SiPixelRecHit* recHit =
           dynamic_cast<const SiPixelRecHit *>(tRecHit);

        if(recHit != 0)
          if(! theFilter->isCompatible(*recHit, gdir, *theCache))
          {
            LogTrace("TrajectFilter")
              << "  [TrajectFilter] fail pixel";
            return false;
          }
      }
      else if(GeomDetEnumerators::isTrackerStrip(ttRecHit->det()->subDetector()))
      { // strip
        if(dynamic_cast<const SiStripMatchedRecHit2D *>(tRecHit)  != 0)
        { // glued
          const SiStripMatchedRecHit2D* recHit =
            dynamic_cast<const SiStripMatchedRecHit2D *>(tRecHit);

          if(recHit != 0)
          { 
            if(! theFilter->isCompatible(recHit->monoHit(), gdir))
            {
              LogTrace("TrajectFilter")
               << "  [TrajectFilter] fail strip matched 1st";
              return false;
            }

            if(! theFilter->isCompatible(recHit->stereoHit(), gdir))
            {
              LogTrace("TrajectFilter")
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
                LogTrace("TrajectFilter")
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
                LogTrace("TrajectFilter")
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
  TempTrajectory t = trajectory;

  // Check if ok
  if(toBeContinued(t)) return true;

  // Should take out last
  if(t.measurements().size() <= 3) return false;

  return true;
}


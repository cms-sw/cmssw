#ifndef TrajectoryStateOnDetInfosTools_H
#define TrajectoryStateOnDetInfosTools_H

#include "RecoTracker/DeDx/interface/TrajectoryStateOnDetInfosProducer.h"

using namespace edm;
using namespace reco;
using namespace std;


namespace TSODI
{
//   TrajectoryStateOnDetInfo* Get_TSODI(const TrajectoryStateOnSurface* trajSOS, TrackingRecHitRef theHit);
   TrackTrajectoryStateOnDetInfosCollection* Fill_TSODICollection(const TrajTrackAssociationCollection TrajToTrackMap, edm::Handle<reco::TrackCollection> trackCollectionHandle);
//   std::auto_ptr<TrackTrajectoryStateOnDetInfosCollection> Fill_TSODICollection(const TrajTrackAssociationCollection TrajToTrackMap, edm::Handle<reco::TrackCollection> trackCollectionHandle);
   void Fill_TSODICollection(const TrajTrackAssociationCollection& TrajToTrackMap, TrackTrajectoryStateOnDetInfosCollection* outputCollection);


   int    clusterSize    (TrajectoryStateOnDetInfo* Tsodi, bool stereo=false);
   int    charge         (TrajectoryStateOnDetInfo* Tsodi, bool stereo=false);
   double thickness      (TrajectoryStateOnDetInfo* Tsodi, edm::ESHandle<TrackerGeometry> tkGeom);
   double pathLength     (TrajectoryStateOnDetInfo* Tsodi, edm::ESHandle<TrackerGeometry> tkGeom);
   float  chargeOverPath (TrajectoryStateOnDetInfo* Tsodi, edm::ESHandle<TrackerGeometry> tkGeom);

}

#endif


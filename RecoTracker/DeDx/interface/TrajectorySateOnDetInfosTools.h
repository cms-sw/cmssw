#ifndef TrajectorySateOnDetInfosTools_H
#define TrajectorySateOnDetInfosTools_H

#include "RecoTracker/DeDx/interface/TrajectorySateOnDetInfosProducer.h"

using namespace edm;
using namespace reco;
using namespace std;


namespace TSODI
{
   TrajectorySateOnDetInfo* Get_TSODI(const Trajectory* traj, const TrajectoryStateOnSurface* trajSOS, const SiStripRecHit2D* hit);
   TrackTrajectorySateOnDetInfosCollection* Get_TSODICollection(const TrajTrackAssociationCollection TrajToTrackMap, edm::Handle<reco::TrackCollection> trackCollectionHandle);
}

#endif


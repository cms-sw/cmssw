#include "DataFormats/TrackReco/interface/TrajectoryStateOnDetInfo.h"

using namespace reco;
using namespace std;

TrajectoryStateOnDetInfo::TrajectoryStateOnDetInfo(const LocalTrajectoryParameters theLocalParameters, std::vector<float> theLocalErrors, const TrackingRecHitRef theTrackingRecHit)
{
   _theLocalParameters = theLocalParameters;
   _theLocalErrors     = theLocalErrors;
   _theTrackingRecHit  = theTrackingRecHit;

}

TrackingRecHitRef TrajectoryStateOnDetInfo::recHit()
{
   return _theTrackingRecHit;
}


LocalVector TrajectoryStateOnDetInfo::momentum(){
   return _theLocalParameters.momentum();
}

LocalPoint TrajectoryStateOnDetInfo::point(){
   return _theLocalParameters.position();
}



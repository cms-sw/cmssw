#include "RecoTracker/CkfPattern/interface/MaxHitsTrajectoryFilter.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoTracker/CkfPattern/interface/TempTrajectory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool MaxHitsTrajectoryFilter::operator()( const TempTrajectory& traj) const 
{
  //theMaxHits < 0 means no cut on number of found hits
  if( (traj.foundHits() < theMaxHits) || theMaxHits<0) return true;
  else return false;
}
bool MaxHitsTrajectoryFilter::operator()( const Trajectory& traj) const 
{
  //theMaxHits < 0 means no cut on number of found hits
  if( (traj.foundHits() < theMaxHits) || theMaxHits<0) return true;
  else return false;
}

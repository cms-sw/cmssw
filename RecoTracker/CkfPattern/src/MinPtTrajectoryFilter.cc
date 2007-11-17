#include "RecoTracker/CkfPattern/interface/MinPtTrajectoryFilter.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoTracker/CkfPattern/interface/TempTrajectory.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateAccessor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


bool MinPtTrajectoryFilter::test( const TrajectoryMeasurement &tm, int foundHits) const 
{
  // check for momentum below limit
  const FreeTrajectoryState& fts = *tm.updatedState().freeTrajectoryState();
  if ( foundHits >= 3 &&
       (
	(1/fts.momentum().perp() > 1/thePtMin + 
	 theNSigma*TrajectoryStateAccessor(fts).inversePtError())
	)
       ||
       ( (fts.momentum().perp()<0.010))
       ||
       ( ( TrajectoryStateAccessor(fts).inversePtError() > 1.e10))
       
       )    return false;
  else return true;
}

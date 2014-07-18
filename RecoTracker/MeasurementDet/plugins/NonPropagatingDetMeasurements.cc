#include "NonPropagatingDetMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"

std::vector<TrajectoryMeasurement> 
NonPropagatingDetMeasurements::get( const MeasurementDet& det,
				    const TrajectoryStateOnSurface& stateOnThisDet,
				    const MeasurementEstimator& est,
                                    const MeasurementTrackerEvent &data) const
{
  throw cms::Exception("THIS SHOULD NOT BE CALLED");
  std::vector<TrajectoryMeasurement> result;
  /*
  MeasurementDet::RecHitContainer allHits = det.recHits(stateOnThisDet, data);
  for (MeasurementDet::RecHitContainer::const_iterator ihit=allHits.begin();
       ihit != allHits.end(); ihit++) {
    std::pair<bool,double> diffEst = est.estimate( stateOnThisDet, **ihit);
    if ( diffEst.first) {
      result.push_back( TrajectoryMeasurement( stateOnThisDet, *ihit, 
					       diffEst.second));
    }
  }
  //GIO// std::cerr << "NonPropagatingDetMeasurements: " << allHits.size() << " => " << result.size() << std::endl;
  if ( result.empty()) {
    // create a TrajectoryMeasurement with an invalid RecHit and zero estimate
    result.push_back( TrajectoryMeasurement( stateOnThisDet, 
					     InvalidTransientRecHit::build(&det.geomDet()), 0.F)); 
  }
  else {
    // sort results according to estimator value
    if ( result.size() > 1) {
      sort( result.begin(), result.end(), TrajMeasLessEstim());
    }
  }
  */
  return result;
}

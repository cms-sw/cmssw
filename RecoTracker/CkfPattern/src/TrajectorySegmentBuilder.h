#ifndef TrajectorySegmentBuilder_H
#define TrajectorySegmentBuilder_H

//B.M. #include "CommonDet/DetUtilities/interface/DetExceptions.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
//B.M.#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include <vector>                                        

#include "FWCore/Utilities/interface/Visibility.h"


class TrajectoryStateUpdator;
class MeasurementEstimator;
class Trajectory;
class TrajectoryMeasurement;
class TrajectoryMeasurementGroup;
class FreeTrajectoryState;
class TrajectoryStateOnSurface;
class DetGroup;
class DetLayer;
class TempTrajectory;

/** 
 */

class  dso_internal TrajectorySegmentBuilder {

private:
  // short names
  typedef FreeTrajectoryState FTS;
  typedef TrajectoryStateOnSurface TSOS;
  typedef TrajectoryMeasurement TM;
  typedef TrajectoryMeasurementGroup TMG;
  typedef std::vector<Trajectory> TrajectoryContainer;
  typedef std::vector<TempTrajectory> TempTrajectoryContainer;
  typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
  
public:
  
  /// constructor from layer and helper objects
  TrajectorySegmentBuilder (const MeasurementTracker* theInputMeasurementTracker,
			    const LayerMeasurements*  theInputLayerMeasurements,
			    const DetLayer& layer,
			    const Propagator& propagator,
			    const TrajectoryStateUpdator& updator,
			    const MeasurementEstimator& estimator,
			    bool lockHits, bool bestHitOnly) :
    theMeasurementTracker(theInputMeasurementTracker),
    theLayerMeasurements(theInputLayerMeasurements),
    theLayer(layer),
    theFullPropagator(propagator),
    theUpdator(updator),
    theEstimator(estimator),
    theGeomPropagator(propagator),
//     theGeomPropagator(propagator.propagationDirection()),
    theLockHits(lockHits),theBestHitOnly(bestHitOnly)
  {}

  /// destructor
  ~TrajectorySegmentBuilder() {}

  /// new segments within layer
  //std::vector<Trajectory> segments (const TSOS startingState);
  TempTrajectoryContainer segments (const TSOS startingState);

private:
  /// update of a trajectory with a hit
  void updateTrajectory (TempTrajectory& traj, const TM& tm) const;
 
 /// creation of new candidates from a segment and a collection of hits
  void updateCandidates (TempTrajectory const& traj, const std::vector<TM>& measurements,
			 TempTrajectoryContainer& candidates);

  /// creation of a new candidate from a segment and the best hit out of a collection
  void updateCandidatesWithBestHit (TempTrajectory const& traj, const std::vector<TM>& measurements,
				    TempTrajectoryContainer& candidates);

  /// retrieve compatible hits from a DetGroup
  std::vector<TrajectoryMeasurement> redoMeasurements (const TempTrajectory& traj,
						  const DetGroup& detGroup) const;

  /// get list of unused hits
  std::vector<TrajectoryMeasurement> unlockedMeasurements (const std::vector<TM>& measurements) const;

  /// mark a hit as used
  void lockMeasurement (const TM& measurement);

  /// clean a set of candidates
  //B.M to be ported later
  void cleanCandidates (std::vector<TempTrajectory>& candidates) const;

// public:

  std::vector<TempTrajectory> addGroup( TempTrajectory const& traj,
			       std::vector<TrajectoryMeasurementGroup>::const_iterator begin,
			       std::vector<TrajectoryMeasurementGroup>::const_iterator end);
    
  void updateWithInvalidHit (TempTrajectory& traj,
			     const std::vector<TMG>& groups,
			     TempTrajectoryContainer& candidates) const;
  
private:
  const MeasurementTracker*     theMeasurementTracker;
  const LayerMeasurements*      theLayerMeasurements;
  const DetLayer&               theLayer;
  const Propagator&             theFullPropagator;
  const TrajectoryStateUpdator& theUpdator;
  const MeasurementEstimator&   theEstimator;
//   AnalyticalPropagator theGeomPropagator;
  const Propagator&             theGeomPropagator;

  bool theLockHits;
  bool theBestHitOnly;
  ConstRecHitContainer theLockedHits;

  bool theDbgFlg;
};

#endif

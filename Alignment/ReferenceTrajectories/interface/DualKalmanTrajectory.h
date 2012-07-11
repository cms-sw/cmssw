#ifndef Alignment_ReferenceTrajectories_DualKalmanTrajectory_H
#define Alignment_ReferenceTrajectories_DualKalmanTrajectory_H

/// \class DualKalmanTrajectory
///
/// A trajectory that provides derivatives w.r.t. to the track parameters as the DualReferenceTrajectory,
/// i.e. extrapolating a helix with five parameters from the hit in the middle.
/// But measurements, trajectory positions and uncertainties are taken from the Kalman track fit.
/// More precisely, the uncertainties are those of the residuals, i.e. of 
/// measurements() - trajectoryPositions().
///
/// Currently three methods to set residual and error can be choosen via cfg:
///   0: the hitError approach, i.e. residual w.r.t. updated state, but error from hit
///   1: the unbiased residal approach as in validation
///   2: the pull approach, i.e. residal w.r.t. to updated state withits error
///
///  \author    : Gero Flucke
///  date       : October 2008
///  $Revision: 1.4 $
///  $Date: 2010/01/14 16:14:03 $
///  (last update by $Author: flucke $)


#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectoryBase.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include <vector>

class ReferenceTrajectory;
namespace reco { class BeamSpot;}

class DualKalmanTrajectory : public ReferenceTrajectoryBase
{

public:

  DualKalmanTrajectory(const Trajectory::DataContainer &trajMeasurements,
		       const TrajectoryStateOnSurface &referenceTsos,
		       const std::vector<unsigned int> &forwardRecHitNums,
		       const std::vector<unsigned int> &backwardRecHitNums,
		       const MagneticField *magField,
		       MaterialEffects materialEffects,
		       PropagationDirection propDir,
		       double mass,
		       bool useBeamSpot,
		       const reco::BeamSpot &beamSpot,
		       int residualMethod);

  virtual ~DualKalmanTrajectory() {}

  virtual DualKalmanTrajectory* clone() const { return new DualKalmanTrajectory(*this); }

protected:

  DualKalmanTrajectory( unsigned int nPar = 0, unsigned int nHits = 0 );

  /// calculate members
  virtual bool construct(const Trajectory::DataContainer &trajMeasurements,
			 const TrajectoryStateOnSurface &referenceTsos,
			 const std::vector<unsigned int> &forwardRecHitNums,
			 const std::vector<unsigned int> &backwardRecHitNums,
			 double mass, MaterialEffects materialEffects,
			 const PropagationDirection propDir, const MagneticField *magField,
			 bool useBeamSpot, const reco::BeamSpot &beamSpot,
			 int residualMethod);

  /// Method to get a single ReferenceTrajectory for a half of the trajectory.
  virtual ReferenceTrajectory* construct(const Trajectory::DataContainer &trajMeasurements,
					 const TrajectoryStateOnSurface &referenceTsos, 
					 const std::vector<unsigned int> &recHits,
					 double mass, MaterialEffects materialEffects,
					 const PropagationDirection propDir,
					 const MagneticField *magField,
					 bool useBeamSpot,
					 const reco::BeamSpot &beamSpot) const;
  /// Fill that part of data members that is different from DualReferenceTrajectory.
  bool fillKalmanPart(const Trajectory::DataContainer &trajMeasurements,
		      const std::vector<unsigned int> &recHitNums, bool startFirst,
		      unsigned int iNextHit, int residualMethod);
  /// helper for 'unbiased residual' method (i.e. returns merged fwd/bwd states)
  TrajectoryStateOnSurface 
  fillMeasurementAndError1(const TransientTrackingRecHit::ConstRecHitPointer &hitPtr,
			   unsigned int iHit, const TrajectoryMeasurement &trajMeasurement);
  /// helper for 'pull' (doPull=true) or 'hitError' method (doPull=false),
  /// returns updated tsos in both cases
  TrajectoryStateOnSurface 
  fillMeasurementAndError2(const TransientTrackingRecHit::ConstRecHitPointer &hitPtr,
			   unsigned int iHit, const TrajectoryMeasurement &trajMeasurement,
			   bool doPull);

  /// fill trajectoryPositions
  void fillTrajectoryPositions(const AlgebraicMatrix &projection,
			       const TrajectoryStateOnSurface &tsos, unsigned int iHit);

  /// local error including APE if APE is on
  LocalError hitErrorWithAPE(const TransientTrackingRecHit::ConstRecHitPointer &hitPtr) const;

  virtual AlgebraicVector extractParameters(const TrajectoryStateOnSurface &referenceTsos) const;

  inline const PropagationDirection oppositeDirection( const PropagationDirection propDir ) const
  { return ((propDir == anyDirection) ? anyDirection 
	    : ((propDir == alongMomentum) ? oppositeToMomentum : alongMomentum));
  }

};

#endif

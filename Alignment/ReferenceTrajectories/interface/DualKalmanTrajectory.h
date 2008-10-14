#ifndef Alignment_ReferenceTrajectories_DualKalmanTrajectory_H
#define Alignment_ReferenceTrajectories_DualKalmanTrajectory_H

/// \class DualKalmanTrajectory
///
/// A trajectory that provides derivatives w.r.t. to the track parameters as the DualReferenceTrajectory,
/// i.e. extrapolating a helix with five parameters from the hit in the middle.
/// But measurements, trajectory positions and uncertainties are taken from the Kalman track fit.
///
/// Currently two methods to set residual and error can be choosen via cfg:
///   1: the unbiased residal approach
///   2: the pull approach
///
///  \author    : Gero Flucke
///  date       : October 2008
///  $Revision$
///  $Date$
///  (last update by $Author$)


#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectoryBase.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include <vector>

class ReferenceTrajectory;
class Trajectory;

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
		       int residualMethod);

  virtual ~DualKalmanTrajectory() {}

  virtual DualKalmanTrajectory* clone() const { return new DualKalmanTrajectory(*this); }

protected:

  DualKalmanTrajectory( unsigned int nPar = 0, unsigned int nHits = 0 );

  /** internal method to calculate members
   */
  virtual bool construct(const Trajectory::DataContainer &trajMeasurements,
			 const TrajectoryStateOnSurface &referenceTsos,
			 const std::vector<unsigned int> &forwardRecHitNums,
			 const std::vector<unsigned int> &backwardRecHitNums,
			 double mass, MaterialEffects materialEffects,
			 const PropagationDirection propDir, const MagneticField *magField,
			 int residualMethod);

  /// Method to get a single ReferenceTrajectory for a half of the trajectory.
  virtual ReferenceTrajectory* construct(const Trajectory::DataContainer &trajMeasurements,
					 const TrajectoryStateOnSurface &referenceTsos, 
					 const std::vector<unsigned int> &recHits,
					 double mass, MaterialEffects materialEffects,
					 const PropagationDirection propDir,
					 const MagneticField *magField) const;
  /// Fill that part of data members that is different from DualReferenceTrajectory.
  bool fillKalmanPart(const Trajectory::DataContainer &trajMeasurements,
		      const std::vector<unsigned int> &recHitNums, bool startFirst,
		      unsigned int iNextHit, int residualMethod);
  /// helper for 'unbiased residual' method
  bool fillMeasurementAndError1(const TransientTrackingRecHit::ConstRecHitPointer &hitPtr,
				unsigned int iHit, const TrajectoryStateOnSurface &tsos);
  /// helper for 'pull' method
  bool fillMeasurementAndError2(const TransientTrackingRecHit::ConstRecHitPointer &hitPtr,
				unsigned int iHit, const TrajectoryStateOnSurface &tsos);
  void fillTrajectoryPositions(const AlgebraicMatrix &projection,
			       const TrajectoryStateOnSurface &tsos, unsigned int iHit);

  virtual AlgebraicVector extractParameters(const TrajectoryStateOnSurface &referenceTsos) const;

  inline const PropagationDirection oppositeDirection( const PropagationDirection propDir ) const
  { return ((propDir == anyDirection) ? anyDirection 
	    : ((propDir == alongMomentum) ? oppositeToMomentum : alongMomentum));
  }

};

#endif

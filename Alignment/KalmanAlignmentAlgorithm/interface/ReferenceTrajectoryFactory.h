#ifndef Alignment_CommonAlignmentAlgorithm_ReferenceTrajectoryFactory_h
#define Alignment_CommonAlignmentAlgorithm_ReferenceTrajectoryFactory_h

#include "Alignment/KalmanAlignmentAlgorithm/interface/TrajectoryFactoryBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/ReferenceTrajectory.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

/// A factory that produces instances of class ReferenceTrajectory from a given TrajTrackPairCollection.


class ReferenceTrajectoryFactory : public TrajectoryFactoryBase
{

public:

  typedef std::pair< TrajectoryStateOnSurface, TransientTrackingRecHit::ConstRecHitContainer > TrajectoryInput;

  ReferenceTrajectoryFactory( const edm::ParameterSet & config );
  virtual ~ReferenceTrajectoryFactory( void );

  /// Produce the reference trajectories.
  virtual const ReferenceTrajectoryCollection trajectories( const edm::EventSetup & setup,
							    const TrajTrackPairCollection & tracks ) const;

  virtual ReferenceTrajectoryFactory* clone( void ) const { return new ReferenceTrajectoryFactory( *this ); }

protected:

  virtual const TrajectoryInput innermostStateAndRecHits( const TrajTrackPair & track ) const;
  virtual const Trajectory::DataContainer orderedTrajectoryMeasurements( const Trajectory & trajectory ) const;

  bool theHitsAreReverse;
  MaterialEffects theMaterialEffects;
  double theMass;
};


#endif

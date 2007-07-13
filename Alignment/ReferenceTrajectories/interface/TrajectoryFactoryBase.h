#ifndef Alignment_ReferenceTrajectories_TrajectoryFactoryBase_h
#define Alignment_ReferenceTrajectories_TrajectoryFactoryBase_h

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectoryBase.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

/// Base class for factories producing reference trajectories, i.e. instances of classes deriving from
/// ReferenceTrajectoryBase, from a TrajTrackPairCollection.


class TrajectoryFactoryBase
{

public:

  typedef ReferenceTrajectoryBase::ReferenceTrajectoryPtr ReferenceTrajectoryPtr;
  typedef ReferenceTrajectoryBase::MaterialEffects MaterialEffects;
  typedef AlignmentAlgorithmBase::ConstTrajTrackPair ConstTrajTrackPair;
  typedef AlignmentAlgorithmBase::ConstTrajTrackPairCollection ConstTrajTrackPairCollection;
  typedef std::vector< ReferenceTrajectoryPtr > ReferenceTrajectoryCollection;
  typedef std::pair< TrajectoryStateOnSurface, TransientTrackingRecHit::ConstRecHitContainer > TrajectoryInput;

  TrajectoryFactoryBase( const edm::ParameterSet & config );
  virtual ~TrajectoryFactoryBase( void );

  virtual const ReferenceTrajectoryCollection trajectories( const edm::EventSetup & setup,
							    const ConstTrajTrackPairCollection & tracks ) const = 0;

  virtual TrajectoryFactoryBase* clone( void ) const = 0;

protected:

  virtual const TrajectoryInput innermostStateAndRecHits( const ConstTrajTrackPair & track ) const;
  virtual const Trajectory::DataContainer orderedTrajectoryMeasurements( const Trajectory & trajectory ) const;

  const MaterialEffects materialEffects( const std::string & strME ) const;

  MaterialEffects theMaterialEffects;
  bool theUseInvalidHits;
};


#endif

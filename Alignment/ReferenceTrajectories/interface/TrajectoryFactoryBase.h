#ifndef Alignment_ReferenceTrajectories_TrajectoryFactoryBase_h
#define Alignment_ReferenceTrajectories_TrajectoryFactoryBase_h

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectoryBase.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

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
  typedef std::vector< TrajectoryStateOnSurface > ExternalPredictionCollection;

  TrajectoryFactoryBase( const edm::ParameterSet & config );
  virtual ~TrajectoryFactoryBase( void );

  virtual const ReferenceTrajectoryCollection trajectories( const edm::EventSetup & setup,
							    const ConstTrajTrackPairCollection & tracks ) const = 0;

  virtual const ReferenceTrajectoryCollection trajectories( const edm::EventSetup& setup,
							    const ConstTrajTrackPairCollection& tracks,
							    const ExternalPredictionCollection& external ) const = 0;

  virtual TrajectoryFactoryBase* clone( void ) const = 0;

  inline const MaterialEffects materialEffects( void ) const { return theMaterialEffects; }
  inline const PropagationDirection propagationDirection( void ) const { return thePropDir; }

protected:

  virtual const TrajectoryInput innermostStateAndRecHits( const ConstTrajTrackPair & track ) const;
  virtual const Trajectory::DataContainer orderedTrajectoryMeasurements( const Trajectory & trajectory ) const;
  bool sameSurface( const Surface& s1, const Surface& s2 ) const;

private:

  bool useRecHit( const TransientTrackingRecHit::ConstRecHitPointer& hitPtr ) const;

  const MaterialEffects materialEffects( const std::string & strME ) const;
  const PropagationDirection propagationDirection( const std::string & strPD ) const;

  MaterialEffects theMaterialEffects;
  PropagationDirection thePropDir;

  bool theUseInvalidHits;
  bool theUseProjectedHits;
};


#endif

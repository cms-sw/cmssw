#ifndef Alignment_ReferenceTrajectories_TrajectoryFactoryBase_h
#define Alignment_ReferenceTrajectories_TrajectoryFactoryBase_h

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectoryBase.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

/// Base class for factories producing reference trajectories, i.e. instances of classes deriving from
/// ReferenceTrajectoryBase, from a TrajTrackPairCollection.


namespace reco { class BeamSpot;}

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

  virtual const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
							   const ConstTrajTrackPairCollection &tracks,
							   const reco::BeamSpot &beamSpot) const = 0;

  virtual const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
							   const ConstTrajTrackPairCollection &tracks,
							   const ExternalPredictionCollection &external,
							   const reco::BeamSpot &beamSpot) const = 0;

  virtual TrajectoryFactoryBase* clone( void ) const = 0;

  inline MaterialEffects materialEffects( void ) const { return materialEffects_; }
  inline PropagationDirection propagationDirection( void ) const { return propDir_; }
  inline const edm::ParameterSet& configuration() const { return cfg_; }

protected:

  virtual const TrajectoryInput innermostStateAndRecHits( const ConstTrajTrackPair & track ) const;
  virtual const Trajectory::DataContainer orderedTrajectoryMeasurements( const Trajectory & trajectory ) const;
  bool sameSurface( const Surface& s1, const Surface& s2 ) const;
  bool useRecHit( const TransientTrackingRecHit::ConstRecHitPointer& hitPtr ) const;

private:

  MaterialEffects materialEffects( const std::string & strME ) const;
  PropagationDirection propagationDirection( const std::string & strPD ) const;

  const edm::ParameterSet cfg_; // need to keep for possible re-use after constructor... :-(
  const MaterialEffects materialEffects_;
  const PropagationDirection propDir_;

  const bool useWithoutDet_;
  const bool useInvalidHits_;
  const bool useProjectedHits_;
  
protected:

  const bool useBeamSpot_;
  const bool includeAPEs_;
};


#endif

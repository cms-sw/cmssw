#ifndef Alignment_KalmanAlignmentAlgorithm_TrajectoryFactoryBase_h
#define Alignment_KalmanAlignmentAlgorithm_TrajectoryFactoryBase_h

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/ReferenceTrajectoryBase.h"

/// Base class for factories producing reference trajectories, i.e. instances of classes deriving from
/// ReferenceTrajectoryBase, from a TrajTrackPairCollection.


class TrajectoryFactoryBase
{

public:

  typedef ReferenceTrajectoryBase::ReferenceTrajectoryPtr ReferenceTrajectoryPtr;
  typedef ReferenceTrajectoryBase::MaterialEffects MaterialEffects;
  typedef AlignmentAlgorithmBase::TrajTrackPair TrajTrackPair;
  typedef AlignmentAlgorithmBase::TrajTrackPairCollection TrajTrackPairCollection;
  typedef std::vector< ReferenceTrajectoryPtr > ReferenceTrajectoryCollection;

  TrajectoryFactoryBase( const edm::ParameterSet & config );
  virtual ~TrajectoryFactoryBase( void );

  virtual const ReferenceTrajectoryCollection trajectories( const edm::EventSetup & setup,
							    const TrajTrackPairCollection & tracks ) const = 0;

  virtual TrajectoryFactoryBase* clone( void ) const = 0;

protected:

  const MaterialEffects materialEffects( const std::string strME ) const;

};


#endif

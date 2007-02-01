#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentUpdator_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentUpdator_h

#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/TrajectoryFactoryBase.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdator.h"

/// Abstract base class for updators for the KalmanAlignmentAlgorithm.


class KalmanAlignmentUpdator
{

public:

  typedef ReferenceTrajectoryBase::ReferenceTrajectoryPtr ReferenceTrajectoryPtr;

  KalmanAlignmentUpdator( const edm::ParameterSet & config ) {}
  virtual ~KalmanAlignmentUpdator( void ) {}

  /// Process some kind of reference trajectory, for instance a single- or two-particle-trajectory,
  /// and calculate an improved estimate on the alignment parameters.
  virtual void process( const ReferenceTrajectoryPtr & trajectory,
			AlignmentParameterStore* store,
			AlignableNavigator* navigator,
			KalmanAlignmentMetricsUpdator* metrics ) = 0;

  virtual KalmanAlignmentUpdator* clone( void ) const = 0;

protected:

  /// Update the AlignmentUserVariables, given that the Alignables hold KalmanAlignmentUserVariables.
  void updateUserVariables( const std::vector< Alignable* > & alignables ) const;

  /// Returns the Alignables associated with the AlignableDets. If two or more AlignableDets are assiocated
  /// to the same Alignable, the Alignable is returned only once.
  const std::vector< Alignable* > alignablesFromAlignableDets( const std::vector< AlignableDet* > alignableDets,
							       AlignmentParameterStore* store ) const;

  const std::vector< AlignableDet* > alignableDetsFromHits( TransientTrackingRecHit::ConstRecHitContainer recHits,
							    AlignableNavigator* navigator ) const;

};


#endif

#ifndef Alignment_KalmanAlignmentAlgorithm_CurrentAlignmentKFUpdator_h
#define Alignment_KalmanAlignmentAlgorithm_CurrentAlignmentKFUpdator_h

/// Update trajectory state by combining predicted state and measurement 
/// as prescribed in the Kalman Filter algorithm plus including the current
/// estimate on the misalignment (if available).

#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"

#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"

class AlignmentParameters;

class CurrentAlignmentKFUpdator : public TrajectoryStateUpdator
{

public:

  CurrentAlignmentKFUpdator( void ) : theAlignableNavigator( 0 ) {}
  CurrentAlignmentKFUpdator( AlignableNavigator* navigator ) : theAlignableNavigator( navigator ) {}
  ~CurrentAlignmentKFUpdator( void ) {}

  template <unsigned int D>
  TrajectoryStateOnSurface update( const TrajectoryStateOnSurface &, const TransientTrackingRecHit & ) const;

  TrajectoryStateOnSurface update( const TrajectoryStateOnSurface &, const TransientTrackingRecHit & ) const;

  virtual CurrentAlignmentKFUpdator * clone( void ) const { return new CurrentAlignmentKFUpdator( *this ); }

private:

  template <unsigned int D>
  void includeCurrentAlignmentEstimate( const TransientTrackingRecHit & aRecHit,
					const TrajectoryStateOnSurface & tsos,
					typename AlgebraicROOTObject<D>::Vector & vecR,
					typename AlgebraicROOTObject<D>::SymMatrix & matV ) const;

  AlignmentParameters* getAlignmentParameters( const AlignableDetOrUnitPtr& alignableDet ) const;
  AlignmentParameters* getHigherLevelParameters( const Alignable* aAlignable ) const;

  AlignableNavigator* theAlignableNavigator;

};

#endif

#ifndef Alignment_KalmanAlignmentAlgorithm_SingleTrajectoryUpdator_h
#define Alignment_KalmanAlignmentAlgorithm_SingleTrajectoryUpdator_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdator.h"

/// A concrete updator for the KalmanAlignmentAlgorithm. It calculates an improved estimate on the
/// current misalignment from a single ReferenceTrajectory.


class CompositeAlignmentParameters;


class SingleTrajectoryUpdator : public KalmanAlignmentUpdator
{

public:

  SingleTrajectoryUpdator( const edm::ParameterSet & config );
  virtual ~SingleTrajectoryUpdator( void );

  /// Calculate the improved estimate.
  virtual void process( const ReferenceTrajectoryPtr & trajectory,
			AlignmentParameterStore* store,
			AlignableNavigator* navigator,
			KalmanAlignmentMetricsUpdator* metrics,
			const MagneticField* magField = 0 );

  virtual SingleTrajectoryUpdator* clone( void ) const { return new SingleTrajectoryUpdator( *this ); }

private:

  bool checkCovariance( const AlgebraicSymMatrix& cov ) const;

  unsigned int theMinNumberOfHits;
  double theExtraWeight;
  double theExternalPredictionWeight;
  bool theCovCheckFlag;

  unsigned int theNumberOfPreAlignmentEvts;
  unsigned int theNumberOfProcessedEvts;
};


#endif

#ifndef Alignment_KalmanAlignmentAlgorithm_DummyUpdator_h
#define Alignment_KalmanAlignmentAlgorithm_DummyUpdator_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdator.h"

/// A dummy alignment-updator for the KalmanAlignmentAlgorithm - it does nothing.

class DummyUpdator : public KalmanAlignmentUpdator
{

public:

  DummyUpdator( const edm::ParameterSet & config );
  virtual ~DummyUpdator( void );

  /// Calculate the improved estimate.
  virtual void process( const ReferenceTrajectoryPtr & trajectory,
			AlignmentParameterStore* store,
			AlignableNavigator* navigator,
			KalmanAlignmentMetricsUpdator* metrics,
			const MagneticField* magField = 0 );

  virtual DummyUpdator* clone( void ) const { return new DummyUpdator( *this ); }

};


#endif

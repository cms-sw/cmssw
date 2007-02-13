
#include "Alignment/KalmanAlignmentAlgorithm/interface/DummyUpdator.h"


DummyUpdator::DummyUpdator( const edm::ParameterSet & config ) : KalmanAlignmentUpdator( config ) {}


DummyUpdator::~DummyUpdator( void ) {}


void DummyUpdator::process( const ReferenceTrajectoryPtr & trajectory,
			    AlignmentParameterStore* store,
			    AlignableNavigator* navigator,
			    KalmanAlignmentMetricsUpdator* metrics ) {}

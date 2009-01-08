
//#include "Alignment/KalmanAlignmentAlgorithm/plugins/DummyUpdator.h"
#include "DummyUpdator.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdatorPlugin.h"


DummyUpdator::DummyUpdator( const edm::ParameterSet & config ) : KalmanAlignmentUpdator( config ) {}


DummyUpdator::~DummyUpdator( void ) {}


void DummyUpdator::process( const ReferenceTrajectoryPtr & trajectory,
			    AlignmentParameterStore* store,
			    AlignableNavigator* navigator,
			    KalmanAlignmentMetricsUpdator* metrics,
			    const MagneticField* magField ) {}

DEFINE_EDM_PLUGIN( KalmanAlignmentUpdatorPlugin, DummyUpdator, "DummyUpdator" );

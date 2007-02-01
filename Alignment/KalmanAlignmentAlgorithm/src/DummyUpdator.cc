
#include "Alignment/KalmanAlignmentAlgorithm/interface/DummyUpdator.h"

#include "Utilities/Timing/interface/TimingReport.h"

#include "CLHEP/Random/RandFlat.h"

#include <set>
#include <iostream>


using namespace std;


DummyUpdator::DummyUpdator( const edm::ParameterSet & config ) :
  KalmanAlignmentUpdator( config )
{}


DummyUpdator::~DummyUpdator( void ) {}


void DummyUpdator::process( const ReferenceTrajectoryPtr & trajectory,
			    AlignmentParameterStore* store,
			    AlignableNavigator* navigator,
			    KalmanAlignmentMetricsUpdator* metrics )
{}

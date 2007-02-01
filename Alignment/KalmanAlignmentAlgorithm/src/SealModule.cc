
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentAlgorithm.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdatorPlugin.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/SingleTrajectoryUpdator.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/DummyUpdator.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdatorPlugin.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/DummyMetricsUpdator.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/SimpleMetricsUpdator.h"

#include "PluginManager/ModuleDef.h"

using namespace edm;

DEFINE_SEAL_MODULE();

// declare the algorithm
DEFINE_SEAL_PLUGIN( AlignmentAlgorithmPluginFactory, KalmanAlignmentAlgorithm, "KalmanAlignmentAlgorithm");

// declare the alignment updators
DEFINE_SEAL_PLUGIN( KalmanAlignmentUpdatorPlugin, SingleTrajectoryUpdator, "SingleTrajectoryUpdator" );
DEFINE_SEAL_PLUGIN( KalmanAlignmentUpdatorPlugin, DummyUpdator, "DummyUpdator" );

// declare the metrics updator
DEFINE_SEAL_PLUGIN( KalmanAlignmentMetricsUpdatorPlugin, DummyMetricsUpdator, "DummyMetricsUpdator" );
DEFINE_SEAL_PLUGIN( KalmanAlignmentMetricsUpdatorPlugin, SimpleMetricsUpdator, "SimpleMetricsUpdator" );

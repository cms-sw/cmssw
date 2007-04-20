
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentAlgorithm.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdatorPlugin.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/SingleTrajectoryUpdator.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/DummyUpdator.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdatorPlugin.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/DummyMetricsUpdator.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/SimpleMetricsUpdator.h"

// declare the algorithm
DEFINE_EDM_PLUGIN( AlignmentAlgorithmPluginFactory, KalmanAlignmentAlgorithm, "KalmanAlignmentAlgorithm");

// declare the alignment updators
DEFINE_EDM_PLUGIN( KalmanAlignmentUpdatorPlugin, SingleTrajectoryUpdator, "SingleTrajectoryUpdator" );
DEFINE_EDM_PLUGIN( KalmanAlignmentUpdatorPlugin, DummyUpdator, "DummyUpdator" );

// declare the metrics updators
DEFINE_EDM_PLUGIN( KalmanAlignmentMetricsUpdatorPlugin, DummyMetricsUpdator, "DummyMetricsUpdator" );
DEFINE_EDM_PLUGIN( KalmanAlignmentMetricsUpdatorPlugin, SimpleMetricsUpdator, "SimpleMetricsUpdator" );

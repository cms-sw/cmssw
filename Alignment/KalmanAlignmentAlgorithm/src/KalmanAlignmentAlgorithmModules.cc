
#include "Alignment/KalmanAlignmentAlgorithm/interface/DataCollector.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentAlgorithm.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdatorPlugin.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/SingleTrajectoryUpdator.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "PluginManager/ModuleDef.h"

using namespace edm;
using namespace serviceregistry;
using namespace alignmentservices;

DEFINE_SEAL_MODULE();
DEFINE_SEAL_PLUGIN( AlignmentAlgorithmPluginFactory, KalmanAlignmentAlgorithm, "KalmanAlignmentAlgorithm");
DEFINE_SEAL_PLUGIN( KalmanAlignmentUpdatorPlugin, SingleTrajectoryUpdator, "SingleTrajectoryUpdator" );
DEFINE_ANOTHER_FWK_SERVICE_MAKER( DataCollector, ParameterSetMaker< DataCollector > )

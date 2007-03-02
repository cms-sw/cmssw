
#include "Alignment/CommonAlignmentAlgorithm/interface/TrajectoryFactoryPlugin.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/ReferenceTrajectoryFactory.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SEAL_MODULE();
DEFINE_SEAL_PLUGIN( TrajectoryFactoryPlugin, ReferenceTrajectoryFactory, "ReferenceTrajectoryFactory" );


#include "Alignment/CommonAlignmentAlgorithm/interface/TrajectoryFactoryPlugin.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/ReferenceTrajectoryFactory.h"

#include "PluginManager/ModuleDef.h"

DEFINE_SEAL_MODULE();
DEFINE_SEAL_PLUGIN( TrajectoryFactoryPlugin, ReferenceTrajectoryFactory, "ReferenceTrajectoryFactory" );

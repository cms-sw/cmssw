// Plugin definition for the algorithm

#include "Alignment/HIPAlignmentAlgorithm/interface/HIPAlignmentAlgorithm.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SEAL_MODULE();
DEFINE_SEAL_PLUGIN( AlignmentAlgorithmPluginFactory,
					HIPAlignmentAlgorithm, "HIPAlignmentAlgorithm" );


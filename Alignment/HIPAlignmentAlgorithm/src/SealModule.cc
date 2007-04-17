// Plugin definition for the algorithm

#include "Alignment/HIPAlignmentAlgorithm/interface/HIPAlignmentAlgorithm.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_EDM_PLUGIN( AlignmentAlgorithmPluginFactory,
		   HIPAlignmentAlgorithm, "HIPAlignmentAlgorithm" );


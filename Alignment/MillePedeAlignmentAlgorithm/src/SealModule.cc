// Plugin definition for the algorithm

#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeAlignmentAlgorithm.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SEAL_MODULE();
DEFINE_SEAL_PLUGIN(AlignmentAlgorithmPluginFactory,
		   MillePedeAlignmentAlgorithm, "MillePedeAlignmentAlgorithm");


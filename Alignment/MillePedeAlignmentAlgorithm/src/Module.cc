// Plugin definition for the algorithm

#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeAlignmentAlgorithm.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_EDM_PLUGIN(AlignmentAlgorithmPluginFactory,
		   MillePedeAlignmentAlgorithm, "MillePedeAlignmentAlgorithm");


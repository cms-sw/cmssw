//<<<<<< INCLUDES                                                       >>>>>>

#include "Geometry/MTCCTrackerCommonData/plugins/DDTIBLayerAlgo_MTCC.h"
#include "Geometry/MTCCTrackerCommonData/plugins/DDTIBRadCableAlgo_MTCC.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDTIBLayerAlgo_MTCC, "track:DDTIBLayerAlgo_MTCC");
DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDTIBRadCableAlgo_MTCC, "track:DDTIBRadCableAlgo_MTCC");

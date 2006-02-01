//<<<<<< INCLUDES                                                       >>>>>>

#include "Geometry/MTCCTrackerCommonData/interface/DDTIBLayerAlgo_MTCC.h"
#include "Geometry/MTCCTrackerCommonData/interface/DDTIBRadCableAlgo_MTCC.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"
#include "PluginManager/ModuleDef.h"

DEFINE_SEAL_MODULE ();
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTIBLayerAlgo_MTCC,    "track:DDTIBLayerAlgo_MTCC");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTIBRadCableAlgo_MTCC, "track:DDTIBRadCableAlgo_MTCC");

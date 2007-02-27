//<<<<<< INCLUDES                                                       >>>>>>

//#include "Geometry/EcalPreshowerData/interface/DDTestAlgorithm.h"
#include "Geometry/EcalTestBeam/interface/DDTBH4Algo.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"
#include "PluginManager/ModuleDef.h"

DEFINE_SEAL_MODULE ();
//DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTestAlgorithm, "DDTestAlgorithm");

DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTBH4Algo, "DDTBH4Algo");

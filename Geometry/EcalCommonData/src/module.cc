//<<<<<< INCLUDES                                                       >>>>>>

//#include "Geometry/EcalPreshowerData/interface/DDTestAlgorithm.h"
#include "Geometry/EcalCommonData/interface/DDEcalBarrelAlgo.h"
#include "Geometry/EcalCommonData/interface/DDEcalPreshowerAlgo.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SEAL_MODULE ();
//DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTestAlgorithm, "DDTestAlgorithm");

DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDEcalBarrelAlgo, "ecal:DDEcalBarrelAlgo");

DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDEcalPreshowerAlgo, "ecal:DDEcalPreshowerAlgo");

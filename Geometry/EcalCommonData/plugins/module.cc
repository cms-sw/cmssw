//<<<<<< INCLUDES                                                       >>>>>>

//#include "Geometry/EcalPreshowerData/interface/DDTestAlgorithm.h"
#include "Geometry/EcalCommonData/interface/DDEcalBarrelAlgo.h"
#include "Geometry/EcalCommonData/interface/DDEcalEndcapAlgo.h"
#include "Geometry/EcalCommonData/interface/DDEcalPreshowerAlgo.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"

//DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTestAlgorithm, "DDTestAlgorithm");

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDEcalEndcapAlgo, "ecal:DDEcalEndcapAlgo");

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDEcalBarrelAlgo, "ecal:DDEcalBarrelAlgo");

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDEcalPreshowerAlgo, "ecal:DDEcalPreshowerAlgo");

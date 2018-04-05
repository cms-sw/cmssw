//<<<<<< INCLUDES                                                       >>>>>>

//#include "Geometry/EcalPreshowerData/interface/DDTestAlgorithm.h"
#include "Geometry/EcalCommonData/interface/DDEcalBarrelAlgo.h"
#include "Geometry/EcalCommonData/interface/DDEcalBarrelNewAlgo.h"
#include "Geometry/EcalCommonData/interface/DDEcalEndcapAlgo.h"
#include "Geometry/EcalCommonData/interface/DDEcalPreshowerAlgo.h"
#include "Geometry/EcalCommonData/interface/DDEcalAPDAlgo.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"

//DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTestAlgorithm, "DDTestAlgorithm");

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDEcalEndcapAlgo, "ecal:DDEcalEndcapAlgo");

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDEcalBarrelAlgo, "ecal:DDEcalBarrelAlgo");

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDEcalBarrelNewAlgo, "ecal:DDEcalBarrelNewAlgo");

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDEcalPreshowerAlgo, "ecal:DDEcalPreshowerAlgo");

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDEcalAPDAlgo, "ecal:DDEcalAPDAlgo");

// #include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Calibration/EcalCalibAlgos/interface/EcalEleCalibLooper.h"
#include "Calibration/EcalCalibAlgos/interface/InvRingCalib.h"
#include "Calibration/EcalCalibAlgos/interface/miscalibExample.h"
#include "Calibration/EcalCalibAlgos/interface/ElectronCalibration.h"
#include "Calibration/EcalCalibAlgos/interface/ZeeCalibration.h"
//#include "Calibration/EcalCalibAlgos/interface/ElectronRecalibSuperClusterAssociator.h"
#include "Calibration/EcalCalibAlgos/interface/PhiSymmetryCalibration.h"
#include "Calibration/EcalCalibAlgos/interface/Pi0FixedMassWindowCalibration.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(miscalibExample);
//DEFINE_ANOTHER_FWK_MODULE(ElectronRecalibSuperClusterAssociator);
DEFINE_ANOTHER_FWK_MODULE(ElectronCalibration);
DEFINE_ANOTHER_FWK_MODULE(PhiSymmetryCalibration);
DEFINE_ANOTHER_FWK_LOOPER(Pi0FixedMassWindowCalibration);
DEFINE_ANOTHER_FWK_LOOPER(ZeeCalibration);
DEFINE_ANOTHER_FWK_LOOPER(EcalEleCalibLooper);	
DEFINE_ANOTHER_FWK_LOOPER(InvRingCalib);


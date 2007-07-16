// #include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Calibration/EcalCalibAlgos/interface/miscalibExample.h"
#include "Calibration/EcalCalibAlgos/interface/ElectronCalibration.h"
#include "Calibration/EcalCalibAlgos/interface/ZeeCalibration.h"
#include "Calibration/EcalCalibAlgos/interface/PhiSymmetryCalibration.h"
#include "Calibration/EcalCalibAlgos/interface/Pi0FixedMassWindowCalibration.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(miscalibExample);
DEFINE_ANOTHER_FWK_MODULE(ElectronCalibration);
DEFINE_ANOTHER_FWK_MODULE(PhiSymmetryCalibration);
DEFINE_ANOTHER_FWK_LOOPER(Pi0FixedMassWindowCalibration);
DEFINE_ANOTHER_FWK_LOOPER(ZeeCalibration);


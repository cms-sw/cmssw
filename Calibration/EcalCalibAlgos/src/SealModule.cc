#include "FWCore/Framework/interface/MakerMacros.h"

#include "Calibration/EcalCalibAlgos/interface/PhiSymmetryCalibration.h"

#include "Calibration/EcalCalibAlgos/src/PhiSymmetryCalibration_step2.h"
#include "Calibration/EcalCalibAlgos/src/PhiSymmetryCalibration_step2_SM.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(PhiSymmetryCalibration);
DEFINE_ANOTHER_FWK_MODULE(PhiSymmetryCalibration_step2);
DEFINE_ANOTHER_FWK_MODULE(PhiSymmetryCalibration_step2_SM);

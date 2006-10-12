#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "CalibMuon/DTCalibration/src/DTTTrigCalibration.h"
#include "CalibMuon/DTCalibration/src/DTT0Calibration.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(DTTTrigCalibration);
DEFINE_ANOTHER_FWK_MODULE(DTT0Calibration);

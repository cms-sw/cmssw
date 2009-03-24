#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMOffline/CalibMuon/interface/DTt0DBValidation.h"
#include "DQMOffline/CalibMuon/interface/DTtTrigDBValidation.h"
#include "DQMOffline/CalibMuon/interface/DTnoiseDBValidation.h"
#include "DQMOffline/CalibMuon/interface/DTPreCalibrationTask.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(DTt0DBValidation);
DEFINE_ANOTHER_FWK_MODULE(DTtTrigDBValidation);
DEFINE_ANOTHER_FWK_MODULE(DTnoiseDBValidation);
DEFINE_ANOTHER_FWK_MODULE(DTPreCalibrationTask);

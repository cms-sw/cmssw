#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "DQMOffline/CalibMuon/interface/DTPreCalibrationTask.h"
#include "DQMOffline/CalibMuon/interface/DTnoiseDBValidation.h"
#include "DQMOffline/CalibMuon/interface/DTt0DBValidation.h"
#include "DQMOffline/CalibMuon/interface/DTtTrigDBValidation.h"

DEFINE_FWK_MODULE(DTt0DBValidation);
DEFINE_FWK_MODULE(DTtTrigDBValidation);
DEFINE_FWK_MODULE(DTnoiseDBValidation);
DEFINE_FWK_MODULE(DTPreCalibrationTask);

#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include <DQM/DTMonitorClient/interface/TestClient.h>
DEFINE_FWK_MODULE(TestClient);

#include <DQM/DTMonitorClient/interface/DTtTrigCalibrationTest.h>
DEFINE_ANOTHER_FWK_MODULE(DTtTrigCalibrationTest);

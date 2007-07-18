#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include <DQM/DTMonitorClient/src/DTtTrigCalibrationTest.h>
DEFINE_FWK_MODULE(DTtTrigCalibrationTest);

#include <DQM/DTMonitorClient/src/DTResolutionTest.h>
DEFINE_ANOTHER_FWK_MODULE(DTResolutionTest);

#include <DQM/DTMonitorClient/src/DTEfficiencyTest.h>
DEFINE_ANOTHER_FWK_MODULE(DTEfficiencyTest);

#include <DQM/DTMonitorClient/src/DTChamberEfficiencyTest.h>
DEFINE_ANOTHER_FWK_MODULE(DTChamberEfficiencyTest);

#include <DQM/DTMonitorClient/src/DTDataIntegrityTest.h>
DEFINE_ANOTHER_FWK_MODULE(DTDataIntegrityTest);

#include "DQM/DTMonitorClient/src/DTNoiseEvaluation.h"
DEFINE_ANOTHER_FWK_MODULE(DTNoiseEvaluation);

#include "DQM/DTMonitorClient/src/DTDeadChannelTest.h"
DEFINE_ANOTHER_FWK_MODULE(DTDeadChannelTest);

#include "DQM/DTMonitorClient/src/DTNoiseTest.h"
DEFINE_ANOTHER_FWK_MODULE(DTNoiseTest);

#include "DQM/DTMonitorClient/src/DTLocalTriggerTest.h"
DEFINE_ANOTHER_FWK_MODULE(DTLocalTriggerTest);

#include "DQM/DTMonitorClient/src/DTCreateSummaryHistos.h"
DEFINE_ANOTHER_FWK_MODULE(DTCreateSummaryHistos);

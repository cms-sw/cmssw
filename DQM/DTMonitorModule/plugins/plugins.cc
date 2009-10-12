#include "FWCore/Framework/interface/MakerMacros.h"

#include <DQM/DTMonitorModule/interface/DTDigiTask.h>
DEFINE_FWK_MODULE(DTDigiTask);

#include <DQM/DTMonitorModule/interface/DTTestPulsesTask.h>
DEFINE_ANOTHER_FWK_MODULE(DTTestPulsesTask);

#include <DQM/DTMonitorModule/src/DTSegmentAnalysisTask.h>
DEFINE_ANOTHER_FWK_MODULE(DTSegmentAnalysisTask);

#include <DQM/DTMonitorModule/src/DTResolutionAnalysisTask.h>
DEFINE_ANOTHER_FWK_MODULE(DTResolutionAnalysisTask);

#include <DQM/DTMonitorModule/src/DTLocalTriggerTask.h>
DEFINE_ANOTHER_FWK_MODULE(DTLocalTriggerTask);

#include <DQM/DTMonitorModule/src/DTEfficiencyTask.h>
DEFINE_ANOTHER_FWK_MODULE(DTEfficiencyTask);

#include <DQM/DTMonitorModule/src/DTChamberEfficiencyTask.h>
DEFINE_ANOTHER_FWK_MODULE(DTChamberEfficiencyTask);

#include <DQM/DTMonitorModule/src/DTChamberEfficiency.h>
DEFINE_ANOTHER_FWK_MODULE(DTChamberEfficiency);

#include "DQM/DTMonitorModule/interface/DTTriggerCheck.h"
DEFINE_ANOTHER_FWK_MODULE(DTTriggerCheck);

#include "DQM/DTMonitorModule/src/DTDigiForNoiseTask.h"
DEFINE_ANOTHER_FWK_MODULE(DTDigiForNoiseTask);

#include "DQM/DTMonitorModule/src/DTNoiseTask.h"
DEFINE_ANOTHER_FWK_MODULE(DTNoiseTask);

#include "DQM/DTMonitorModule/src/DTAlbertoBenvenutiTask.h"
DEFINE_ANOTHER_FWK_MODULE(DTAlbertoBenvenutiTask);

#include "DQM/DTMonitorModule/interface/DTCalibValidation.h"
DEFINE_ANOTHER_FWK_MODULE(DTCalibValidation);

#include "DQM/DTMonitorModule/src/DTROMonitorFilter.h"
DEFINE_ANOTHER_FWK_MODULE(DTROMonitorFilter);

#include <DQM/DTMonitorModule/src/DTTriggerEfficiencyTask.h>
DEFINE_ANOTHER_FWK_MODULE(DTTriggerEfficiencyTask);

#include <DQM/DTMonitorModule/interface/DTDataIntegrityTask.h>
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using namespace edm::serviceregistry;

//typedef ParameterSetMaker<DTDataMonitorInterface,DTDataIntegrityTask> maker;
typedef edm::serviceregistry::AllArgsMaker<DTDataMonitorInterface,DTDataIntegrityTask> maker;

DEFINE_ANOTHER_FWK_SERVICE_MAKER(DTDataIntegrityTask,maker);

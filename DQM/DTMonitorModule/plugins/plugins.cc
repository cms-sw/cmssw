#include "FWCore/Framework/interface/MakerMacros.h"

#include <DQM/DTMonitorModule/interface/DTDigiTask.h>
DEFINE_FWK_MODULE(DTDigiTask);

#include <DQM/DTMonitorModule/src/DTSegmentAnalysisTask.h>
DEFINE_FWK_MODULE(DTSegmentAnalysisTask);

#include <DQM/DTMonitorModule/src/DTResolutionAnalysisTask.h>
DEFINE_FWK_MODULE(DTResolutionAnalysisTask);

#include <DQM/DTMonitorModule/src/DTLocalTriggerTask.h>
DEFINE_FWK_MODULE(DTLocalTriggerTask);

#include <DQM/DTMonitorModule/src/DTEfficiencyTask.h>
DEFINE_FWK_MODULE(DTEfficiencyTask);

#include <DQM/DTMonitorModule/src/DTChamberEfficiencyTask.h>
DEFINE_FWK_MODULE(DTChamberEfficiencyTask);

#include <DQM/DTMonitorModule/src/DTChamberEfficiency.h>
DEFINE_FWK_MODULE(DTChamberEfficiency);

#include <DQM/DTMonitorModule/src/DTRunConditionVar.h>
DEFINE_FWK_MODULE(DTRunConditionVar);

#include "DQM/DTMonitorModule/src/DTNoiseTask.h"
DEFINE_FWK_MODULE(DTNoiseTask);

#include "DQM/DTMonitorModule/interface/DTCalibValidation.h"
DEFINE_FWK_MODULE(DTCalibValidation);

#include <DQM/DTMonitorModule/src/DTTriggerEfficiencyTask.h>
DEFINE_FWK_MODULE(DTTriggerEfficiencyTask);

#include <DQM/DTMonitorModule/src/DTLocalTriggerSynchTask.h>
DEFINE_FWK_MODULE(DTLocalTriggerSynchTask);

#include <DQM/DTMonitorModule/src/DTLocalTriggerLutTask.h>
DEFINE_FWK_MODULE(DTLocalTriggerLutTask);

#include <DQM/DTMonitorModule/src/DTLocalTriggerBaseTask.h>
DEFINE_FWK_MODULE(DTLocalTriggerBaseTask);

#include <DQM/DTMonitorModule/src/DTScalerInfoTask.h>
DEFINE_FWK_MODULE(DTScalerInfoTask);

#include <DQM/DTMonitorModule/src/DTDCSByLumiTask.h>
DEFINE_FWK_MODULE(DTDCSByLumiTask);


#include <DQM/DTMonitorModule/interface/DTDataIntegrityTask.h>
DEFINE_FWK_MODULE(DTDataIntegrityTask);

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:

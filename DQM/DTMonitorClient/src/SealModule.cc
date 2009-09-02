
#include "FWCore/Framework/interface/MakerMacros.h"

#include <DQM/DTMonitorClient/src/DTtTrigCalibrationTest.h>
DEFINE_FWK_MODULE(DTtTrigCalibrationTest);

#include <DQM/DTMonitorClient/src/DTResolutionTest.h>
DEFINE_ANOTHER_FWK_MODULE(DTResolutionTest);

#include <DQM/DTMonitorClient/src/DTEfficiencyTest.h>
DEFINE_ANOTHER_FWK_MODULE(DTEfficiencyTest);

#include <DQM/DTMonitorClient/src/DTChamberEfficiencyTest.h>
DEFINE_ANOTHER_FWK_MODULE(DTChamberEfficiencyTest);

#include <DQM/DTMonitorClient/src/DTChamberEfficiencyClient.h>
DEFINE_ANOTHER_FWK_MODULE(DTChamberEfficiencyClient);

#include <DQM/DTMonitorClient/src/DTDataIntegrityTest.h>
DEFINE_ANOTHER_FWK_MODULE(DTDataIntegrityTest);

#include <DQM/DTMonitorClient/src/DTSegmentAnalysisTest.h>
DEFINE_ANOTHER_FWK_MODULE(DTSegmentAnalysisTest);

#include "DQM/DTMonitorClient/src/DTDeadChannelTest.h"
DEFINE_ANOTHER_FWK_MODULE(DTDeadChannelTest);

#include "DQM/DTMonitorClient/src/DTNoiseTest.h"
DEFINE_ANOTHER_FWK_MODULE(DTNoiseTest);

#include "DQM/DTMonitorClient/src/DTNoiseAnalysisTest.h"
DEFINE_ANOTHER_FWK_MODULE(DTNoiseAnalysisTest);

#include "DQM/DTMonitorClient/src/DTLocalTriggerTest.h"
DEFINE_ANOTHER_FWK_MODULE(DTLocalTriggerTest);

#include "DQM/DTMonitorClient/src/DTLocalTriggerEfficiencyTest.h"
DEFINE_ANOTHER_FWK_MODULE(DTLocalTriggerEfficiencyTest);

#include "DQM/DTMonitorClient/src/DTLocalTriggerLutTest.h"
DEFINE_ANOTHER_FWK_MODULE(DTLocalTriggerLutTest);

#include "DQM/DTMonitorClient/src/DTLocalTriggerTPTest.h"
DEFINE_ANOTHER_FWK_MODULE(DTLocalTriggerTPTest);

#include "DQM/DTMonitorClient/src/DTCreateSummaryHistos.h"
DEFINE_ANOTHER_FWK_MODULE(DTCreateSummaryHistos);

#include "DQM/DTMonitorClient/src/DTOccupancyTest.h"
DEFINE_ANOTHER_FWK_MODULE(DTOccupancyTest);

#include "DQM/DTMonitorClient/src/DTSummaryClients.h"
DEFINE_ANOTHER_FWK_MODULE(DTSummaryClients);

#include "DQM/DTMonitorClient/src/DTOfflineSummaryClients.h"
DEFINE_ANOTHER_FWK_MODULE(DTOfflineSummaryClients);

#include <DQM/DTMonitorClient/src/DTResolutionAnalysisTest.h>
DEFINE_ANOTHER_FWK_MODULE(DTResolutionAnalysisTest);

#include <DQM/DTMonitorClient/src/DTDAQInfo.h>
DEFINE_ANOTHER_FWK_MODULE(DTDAQInfo);

#include <DQM/DTMonitorClient/src/DTDCSSummary.h>
DEFINE_ANOTHER_FWK_MODULE(DTDCSSummary);

#include <DQM/DTMonitorClient/src/DTCertificationSummary.h>
DEFINE_ANOTHER_FWK_MODULE(DTCertificationSummary);

#include "DQM/DTMonitorClient/src/DTTriggerEfficiencyTest.h"
DEFINE_ANOTHER_FWK_MODULE(DTTriggerEfficiencyTest);



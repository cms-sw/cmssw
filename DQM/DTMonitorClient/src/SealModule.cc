
#include "FWCore/Framework/interface/MakerMacros.h"

#include <DQM/DTMonitorClient/src/DTResolutionTest.h>
DEFINE_FWK_MODULE(DTResolutionTest);

#include <DQM/DTMonitorClient/src/DTEfficiencyTest.h>
DEFINE_FWK_MODULE(DTEfficiencyTest);

#include <DQM/DTMonitorClient/src/DTChamberEfficiencyTest.h>
DEFINE_FWK_MODULE(DTChamberEfficiencyTest);

#include <DQM/DTMonitorClient/src/DTChamberEfficiencyClient.h>
DEFINE_FWK_MODULE(DTChamberEfficiencyClient);

#include <DQM/DTMonitorClient/src/DTDataIntegrityTest.h>
DEFINE_FWK_MODULE(DTDataIntegrityTest);

#include <DQM/DTMonitorClient/src/DTSegmentAnalysisTest.h>
DEFINE_FWK_MODULE(DTSegmentAnalysisTest);

#include "DQM/DTMonitorClient/src/DTNoiseAnalysisTest.h"
DEFINE_FWK_MODULE(DTNoiseAnalysisTest);

#include "DQM/DTMonitorClient/src/DTLocalTriggerTest.h"
DEFINE_FWK_MODULE(DTLocalTriggerTest);

#include "DQM/DTMonitorClient/src/DTLocalTriggerEfficiencyTest.h"
DEFINE_FWK_MODULE(DTLocalTriggerEfficiencyTest);

#include "DQM/DTMonitorClient/src/DTLocalTriggerLutTest.h"
DEFINE_FWK_MODULE(DTLocalTriggerLutTest);

#include "DQM/DTMonitorClient/src/DTLocalTriggerSynchTest.h"
DEFINE_FWK_MODULE(DTLocalTriggerSynchTest);

#include "DQM/DTMonitorClient/src/DTLocalTriggerTPTest.h"
DEFINE_FWK_MODULE(DTLocalTriggerTPTest);

#include "DQM/DTMonitorClient/src/DTOccupancyTest.h"
DEFINE_FWK_MODULE(DTOccupancyTest);

#include "DQM/DTMonitorClient/src/DTSummaryClients.h"
DEFINE_FWK_MODULE(DTSummaryClients);

#include "DQM/DTMonitorClient/src/DTOfflineSummaryClients.h"
DEFINE_FWK_MODULE(DTOfflineSummaryClients);

#include <DQM/DTMonitorClient/src/DTResolutionAnalysisTest.h>
DEFINE_FWK_MODULE(DTResolutionAnalysisTest);

#include <DQM/DTMonitorClient/src/DTDAQInfo.h>
DEFINE_FWK_MODULE(DTDAQInfo);

#include <DQM/DTMonitorClient/src/DTDCSSummary.h>
DEFINE_FWK_MODULE(DTDCSSummary);

#include <DQM/DTMonitorClient/src/DTDCSByLumiSummary.h>
DEFINE_FWK_MODULE(DTDCSByLumiSummary);

#include <DQM/DTMonitorClient/src/DTCertificationSummary.h>
DEFINE_FWK_MODULE(DTCertificationSummary);

#include "DQM/DTMonitorClient/src/DTTriggerEfficiencyTest.h"
DEFINE_FWK_MODULE(DTTriggerEfficiencyTest);

#include "DQM/DTMonitorClient/src/DTTriggerLutTest.h"
DEFINE_FWK_MODULE(DTTriggerLutTest);

#include "DQM/DTMonitorClient/src/DTRunConditionVarClient.h"
DEFINE_FWK_MODULE(DTRunConditionVarClient);

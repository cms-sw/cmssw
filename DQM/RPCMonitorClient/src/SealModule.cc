#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

//#include <DQM/RPCMonitorClient/interface/RPCDeadChannelTest.h>
#include <DQM/RPCMonitorClient/interface/RPCQualityTests.h>
#include <DQM/RPCMonitorClient/interface/RPCEventSummary.h>
#include <DQM/RPCMonitorClient/interface/RPCTriggerFilter.h>

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(RPCQualityTests);
DEFINE_ANOTHER_FWK_MODULE(RPCEventSummary);
DEFINE_ANOTHER_FWK_MODULE(RPCTriggerFilter);


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();
#include <DQM/RPCMonitorClient/interface/RPCDeadChannelTest.h>
DEFINE_ANOTHER_FWK_MODULE(RPCDeadChannelTest);
#include <DQM/RPCMonitorClient/interface/RPCEventSummary.h>
DEFINE_ANOTHER_FWK_MODULE(RPCEventSummary);
#include <DQM/RPCMonitorClient/interface/RPCMon_SS_Dbx_Global.h>
DEFINE_ANOTHER_FWK_MODULE(RPCMon_SS_Dbx_Global);
#include <DQM/RPCMonitorClient/interface/RPCFEDIntegrity.h>
DEFINE_ANOTHER_FWK_MODULE(RPCFEDIntegrity);
#include <DQM/RPCMonitorClient/interface/RPCMonitorRaw.h>
DEFINE_ANOTHER_FWK_MODULE(RPCMonitorRaw);

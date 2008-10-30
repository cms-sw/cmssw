#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <DQM/RPCMonitorClient/interface/RPCDeadChannelTest.h>
#include <DQM/RPCMonitorClient/interface/RPCEventSummary.h>
#include <DQM/RPCMonitorClient/interface/RPCMon_SS_Dbx_Global.h>

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RPCDeadChannelTest);
DEFINE_ANOTHER_FWK_MODULE(RPCEventSummary);
DEFINE_ANOTHER_FWK_MODULE(RPCMon_SS_Dbx_Global);


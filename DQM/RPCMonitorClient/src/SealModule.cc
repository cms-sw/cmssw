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
#include <DQM/RPCMonitorClient/interface/RPCDaqInfo.h>
DEFINE_ANOTHER_FWK_MODULE(RPCDaqInfo);
#include <DQM/RPCMonitorClient/interface/RPCOccupancyTest.h>
DEFINE_ANOTHER_FWK_MODULE(RPCOccupancyTest);
#include <DQM/RPCMonitorClient/interface/RPCClusterSizeTest.h>
DEFINE_ANOTHER_FWK_MODULE(RPCClusterSizeTest);
#include <DQM/RPCMonitorClient/interface/ReadMeFromFile.h>
DEFINE_ANOTHER_FWK_MODULE(ReadMeFromFile);
#include <DQM/RPCMonitorClient/interface/RPCChamberQuality.h>
DEFINE_ANOTHER_FWK_MODULE(RPCChamberQuality);
// #include <DQM/RPCMonitorClient/interface/RPCDCSDataSimulator.h>
// DEFINE_ANOTHER_FWK_MODULE(RPCDCSDataSimulator);
#include <DQM/RPCMonitorClient/interface/RPCMultiplicityTest.h>
DEFINE_ANOTHER_FWK_MODULE(RPCMultiplicityTest);
#include <DQM/RPCMonitorClient/interface/RPCOccupancyChipTest.h>
DEFINE_ANOTHER_FWK_MODULE(RPCOccupancyChipTest);

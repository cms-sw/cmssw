#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

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


//Used to read ME from ROOT files
#include <DQM/RPCMonitorClient/interface/ReadMeFromFile.h>
DEFINE_ANOTHER_FWK_MODULE(ReadMeFromFile);

//General Client
#include <DQM/RPCMonitorClient/interface/RPCDqmClient.h>
DEFINE_ANOTHER_FWK_MODULE(RPCDqmClient);


#include <DQM/RPCMonitorClient/interface/RPCChamberQuality.h>
DEFINE_ANOTHER_FWK_MODULE(RPCChamberQuality);

// #include <DQM/RPCMonitorClient/interface/RPCDCSDataSimulator.h>
// DEFINE_ANOTHER_FWK_MODULE(RPCDCSDataSimulator);


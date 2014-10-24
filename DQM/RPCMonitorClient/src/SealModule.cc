#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <DQM/RPCMonitorClient/interface/RPCFEDIntegrity.h>
DEFINE_FWK_MODULE(RPCFEDIntegrity);

#include <DQM/RPCMonitorClient/interface/RPCMonitorRaw.h>
DEFINE_FWK_MODULE(RPCMonitorRaw);

#include <DQM/RPCMonitorClient/interface/RPCMonitorLinkSynchro.h>
DEFINE_FWK_MODULE(RPCMonitorLinkSynchro);


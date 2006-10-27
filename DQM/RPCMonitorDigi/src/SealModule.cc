#include "DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h"
#include "DQM/RPCMonitorDigi/interface/RPCMonitorSync.h"
#include "DQM/RPCMonitorDigi/interface/RPCMonitorEfficiency.h"
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RPCMonitorDigi);
DEFINE_ANOTHER_FWK_MODULE(RPCMonitorSync);
DEFINE_ANOTHER_FWK_MODULE(RPCMonitorEfficiency);


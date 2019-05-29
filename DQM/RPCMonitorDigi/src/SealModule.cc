#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h"
DEFINE_FWK_MODULE(RPCMonitorDigi);
#include "DQM/RPCMonitorDigi/interface/RPCRecHitProbability.h"
DEFINE_FWK_MODULE(RPCRecHitProbability);
#include "DQM/RPCMonitorDigi/interface/RPCTTUMonitor.h"
DEFINE_FWK_MODULE(RPCTTUMonitor);
#include "DQM/RPCMonitorDigi/interface/RPCDcsInfo.h"
DEFINE_FWK_MODULE(RPCDcsInfo);

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include <DQM/L1TMonitorClient/interface/L1TCaloClient.h>
DEFINE_ANOTHER_FWK_MODULE(L1TCaloClient);

#include <DQM/L1TMonitorClient/interface/L1TMuonClient.h>
DEFINE_ANOTHER_FWK_MODULE(L1TMuonClient);

#include <DQM/L1TMonitorClient/interface/L1TDTTPGClient.h>
DEFINE_ANOTHER_FWK_MODULE(L1TDTTPGClient);

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <DQM/L1TClient/interface/L1TCaloClient.h>
#include <DQM/L1TClient/interface/L1TMuonClient.h>
DEFINE_FWK_MODULE(L1TCaloClient);
DEFINE_ANOTHER_FWK_MODULE(L1TMuonClient);

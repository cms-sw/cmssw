#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMOffline/Ecal/interface/EcalOfflineCosmicTask.h"
#include "DQMOffline/Ecal/interface/EcalOfflineCosmicClient.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(EcalOfflineCosmicTask);
DEFINE_ANOTHER_FWK_MODULE(EcalOfflineCosmicClient);

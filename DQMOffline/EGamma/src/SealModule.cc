#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMOffline/EGamma/interface/PhotonAnalyzer.h"
#include "DQMOffline/EGamma/interface/ElectronAnalyzer.h"
#include "DQMOffline/EGamma/interface/PhotonOfflineClient.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(PhotonAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(ElectronAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(PhotonOfflineClient);

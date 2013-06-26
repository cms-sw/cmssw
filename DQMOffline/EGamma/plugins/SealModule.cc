#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMOffline/EGamma/plugins/PhotonAnalyzer.h"
#include "DQMOffline/EGamma/plugins/ZToMuMuGammaAnalyzer.h"
#include "DQMOffline/EGamma/plugins/PhotonOfflineClient.h"
#include "DQMOffline/EGamma/plugins/PhotonDataCertification.h"
#include "DQMOffline/EGamma/plugins/PiZeroAnalyzer.h"
#include "DQMOffline/EGamma/plugins/ElectronGeneralAnalyzer.h"
#include "DQMOffline/EGamma/plugins/ElectronAnalyzer.h"
#include "DQMOffline/EGamma/plugins/ElectronTagProbeAnalyzer.h"
#include "DQMOffline/EGamma/plugins/ElectronOfflineClient.h"




DEFINE_FWK_MODULE(PhotonAnalyzer);
DEFINE_FWK_MODULE(ZToMuMuGammaAnalyzer);
DEFINE_FWK_MODULE(PhotonOfflineClient);
DEFINE_FWK_MODULE(PhotonDataCertification);
DEFINE_FWK_MODULE(PiZeroAnalyzer);
DEFINE_FWK_MODULE(ElectronGeneralAnalyzer);
DEFINE_FWK_MODULE(ElectronAnalyzer);
DEFINE_FWK_MODULE(ElectronTagProbeAnalyzer);
DEFINE_FWK_MODULE(ElectronOfflineClient);

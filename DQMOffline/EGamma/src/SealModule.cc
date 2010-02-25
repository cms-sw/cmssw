#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMOffline/EGamma/interface/PhotonAnalyzer.h"
#include "DQMOffline/EGamma/interface/PhotonOfflineClient.h"
#include "DQMOffline/EGamma/interface/PhotonDataCertification.h"
#include "DQMOffline/EGamma/interface/PiZeroAnalyzer.h"
#include "DQMOffline/EGamma/interface/ElectronGeneralAnalyzer.h"
#include "DQMOffline/EGamma/interface/ElectronAnalyzer.h"
#include "DQMOffline/EGamma/interface/ElectronTagProbeAnalyzer.h"
#include "DQMOffline/EGamma/interface/ElectronOfflineClient.h"




DEFINE_FWK_MODULE(PhotonAnalyzer);
DEFINE_FWK_MODULE(PhotonOfflineClient);
DEFINE_FWK_MODULE(PhotonDataCertification);
DEFINE_FWK_MODULE(PiZeroAnalyzer);
DEFINE_FWK_MODULE(ElectronGeneralAnalyzer);
DEFINE_FWK_MODULE(ElectronAnalyzer);
DEFINE_FWK_MODULE(ElectronTagProbeAnalyzer);
DEFINE_FWK_MODULE(ElectronOfflineClient);

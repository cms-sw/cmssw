#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "RecoEgamma/Examples/interface/SiStripElectronAnalyzer.h"
#include "RecoEgamma/Examples/interface/ElectronAnalyzer.h"
#include "RecoEgamma/Examples/interface/ElectronPixelSeedAnalyzer.h"
#include "RecoEgamma/Examples/interface/PixelMatchElectronAnalyzer.h"
#include "RecoEgamma/Examples/interface/PixelMatchGsfElectronAnalyzer.h"
#include "RecoEgamma/Examples/interface/SimplePhotonAnalyzer.h"
#include "RecoEgamma/Examples/interface/SimpleConvertedPhotonAnalyzer.h"
#include "RecoEgamma/Examples/interface/ElectronIDAnalyzer.h"
#include "RecoEgamma/Examples/interface/MCPhotonAnalyzer.h"
#include "RecoEgamma/Examples/interface/MCElectronAnalyzer.h"
#include "RecoEgamma/Examples/interface/MCPizeroAnalyzer.h"


//DEFINE_SEAL_MODULE();

DEFINE_FWK_MODULE(SiStripElectronAnalyzer);
DEFINE_FWK_MODULE(ElectronAnalyzer);
DEFINE_FWK_MODULE(PixelMatchElectronAnalyzer);
DEFINE_FWK_MODULE(PixelMatchGsfElectronAnalyzer);
DEFINE_FWK_MODULE(ElectronPixelSeedAnalyzer);
DEFINE_FWK_MODULE(SimplePhotonAnalyzer);
DEFINE_FWK_MODULE(SimpleConvertedPhotonAnalyzer);
DEFINE_FWK_MODULE(ElectronIDAnalyzer);
DEFINE_FWK_MODULE(MCPhotonAnalyzer);
DEFINE_FWK_MODULE(MCElectronAnalyzer);
DEFINE_FWK_MODULE(MCPizeroAnalyzer);

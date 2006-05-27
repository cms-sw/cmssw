#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoEgamma/EgammaElectronProducers/interface/SiStripElectronProducer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronAnalyzer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(SiStripElectronProducer);
DEFINE_ANOTHER_FWK_MODULE(ElectronAnalyzer);

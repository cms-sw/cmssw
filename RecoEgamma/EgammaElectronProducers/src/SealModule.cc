#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoEgamma/EgammaElectronProducers/interface/SiStripElectronProducer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/SiStripElectronAssociator.h"
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronProducer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronAnalyzer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronPixelSeedProducer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronPixelSeedAnalyzer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/PixelMatchElectronProducer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/PixelMatchElectronAnalyzer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(ElectronProducer);
DEFINE_ANOTHER_FWK_MODULE(SiStripElectronProducer);
DEFINE_ANOTHER_FWK_MODULE(SiStripElectronAssociator);
DEFINE_ANOTHER_FWK_MODULE(ElectronAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(ElectronPixelSeedProducer)
DEFINE_ANOTHER_FWK_MODULE(PixelMatchElectronProducer)
DEFINE_ANOTHER_FWK_MODULE(PixelMatchElectronAnalyzer)
DEFINE_ANOTHER_FWK_MODULE(ElectronPixelSeedAnalyzer)

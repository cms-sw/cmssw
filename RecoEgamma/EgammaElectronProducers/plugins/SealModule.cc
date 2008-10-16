#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectronFwd.h"
#include "SiStripElectronProducer.h"
#include "SiStripElectronAssociator.h"
#include "ElectronPixelSeedProducer.h"
#include "GlobalSeedProducer.h"
#include "GsfElectronProducer.h"
#include "GlobalGsfElectronProducer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(SiStripElectronProducer);
DEFINE_ANOTHER_FWK_MODULE(SiStripElectronAssociator);
DEFINE_ANOTHER_FWK_MODULE(ElectronPixelSeedProducer);
DEFINE_ANOTHER_FWK_MODULE(GlobalSeedProducer);
DEFINE_ANOTHER_FWK_MODULE(GsfElectronProducer);
DEFINE_ANOTHER_FWK_MODULE(GlobalGsfElectronProducer);

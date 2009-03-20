#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectronFwd.h"
#include "SiStripElectronProducer.h"
#include "SiStripElectronAssociator.h"
#include "ElectronSeedProducer.h"
#include "SiStripElectronSeedProducer.h"
//#include "GlobalSeedProducer.h"
#include "GsfElectronCoreProducer.h"
#include "GsfElectronProducer.h"
//#include "GlobalGsfElectronProducer.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(SiStripElectronProducer);
DEFINE_ANOTHER_FWK_MODULE(SiStripElectronAssociator);
DEFINE_ANOTHER_FWK_MODULE(ElectronSeedProducer);
//DEFINE_ANOTHER_FWK_MODULE(GlobalSeedProducer);
DEFINE_ANOTHER_FWK_MODULE(GsfElectronCoreProducer);
DEFINE_ANOTHER_FWK_MODULE(GsfElectronProducer);
//DEFINE_ANOTHER_FWK_MODULE(GlobalGsfElectronProducer);
DEFINE_ANOTHER_FWK_MODULE(SiStripElectronSeedProducer);

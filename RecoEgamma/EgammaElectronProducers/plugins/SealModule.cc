#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectronFwd.h"
#include "SiStripElectronProducer.h"
#include "SiStripElectronAssociator.h"
#include "ElectronSeedProducer.h"
#include "SiStripElectronSeedProducer.h"
//#include "GlobalSeedProducer.h"
#include "GsfElectronCoreEcalDrivenProducer.h"
#include "GsfElectronEcalDrivenProducer.h"
#include "GsfElectronCoreProducer.h"
#include "GsfElectronProducer.h"
//#include "GlobalGsfElectronProducer.h"

#include "GEDGsfElectronCoreProducer.h"
#include "GEDGsfElectronProducer.h"
#include "GEDGsfElectronFinalizer.h"

DEFINE_FWK_MODULE(SiStripElectronProducer);
DEFINE_FWK_MODULE(SiStripElectronAssociator);
DEFINE_FWK_MODULE(ElectronSeedProducer);
//DEFINE_FWK_MODULE(GlobalSeedProducer);
DEFINE_FWK_MODULE(GsfElectronCoreEcalDrivenProducer);
DEFINE_FWK_MODULE(GsfElectronEcalDrivenProducer);
DEFINE_FWK_MODULE(GsfElectronCoreProducer);
DEFINE_FWK_MODULE(GsfElectronProducer);
//DEFINE_FWK_MODULE(GlobalGsfElectronProducer);
DEFINE_FWK_MODULE(SiStripElectronSeedProducer);
DEFINE_FWK_MODULE(GEDGsfElectronCoreProducer);
DEFINE_FWK_MODULE(GEDGsfElectronProducer);
DEFINE_FWK_MODULE(GEDGsfElectronFinalizer);

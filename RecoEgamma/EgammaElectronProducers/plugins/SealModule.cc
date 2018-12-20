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
#include "GsfElectronFull5x5Filler.h"
//#include "GlobalGsfElectronProducer.h"

#include "GEDGsfElectronCoreProducer.h"
#include "GEDGsfElectronProducer.h"
#include "GEDGsfElectronFinalizer.h"

#include "LowPtGsfElectronCoreProducer.h"
#include "LowPtGsfElectronIDProducer.h"
#include "LowPtGsfElectronProducer.h"
#include "LowPtGsfElectronSeedProducer.h"
#include "LowPtGsfElectronSeedValueMapsProducer.h"
#include "LowPtGsfElectronSCProducer.h"

DEFINE_FWK_MODULE(SiStripElectronProducer);
DEFINE_FWK_MODULE(SiStripElectronAssociator);
DEFINE_FWK_MODULE(ElectronSeedProducer);
//DEFINE_FWK_MODULE(GlobalSeedProducer);
DEFINE_FWK_MODULE(GsfElectronFull5x5Filler);
DEFINE_FWK_MODULE(GsfElectronCoreEcalDrivenProducer);
DEFINE_FWK_MODULE(GsfElectronEcalDrivenProducer);
DEFINE_FWK_MODULE(GsfElectronCoreProducer);
DEFINE_FWK_MODULE(GsfElectronProducer);
//DEFINE_FWK_MODULE(GlobalGsfElectronProducer);
DEFINE_FWK_MODULE(SiStripElectronSeedProducer);
DEFINE_FWK_MODULE(GEDGsfElectronCoreProducer);
DEFINE_FWK_MODULE(GEDGsfElectronProducer);
DEFINE_FWK_MODULE(GEDGsfElectronFinalizer);
DEFINE_FWK_MODULE(LowPtGsfElectronCoreProducer);
DEFINE_FWK_MODULE(LowPtGsfElectronIDProducer);
DEFINE_FWK_MODULE(LowPtGsfElectronProducer);
DEFINE_FWK_MODULE(LowPtGsfElectronSeedProducer);
DEFINE_FWK_MODULE(LowPtGsfElectronSeedValueMapsProducer);
DEFINE_FWK_MODULE(LowPtGsfElectronSCProducer);

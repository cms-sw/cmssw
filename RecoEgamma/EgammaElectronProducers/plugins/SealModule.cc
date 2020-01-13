#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "GsfElectronCoreEcalDrivenProducer.h"

#include "GEDGsfElectronCoreProducer.h"
#include "GEDGsfElectronProducer.h"
#include "GEDGsfElectronFinalizer.h"

DEFINE_FWK_MODULE(GsfElectronCoreEcalDrivenProducer);
DEFINE_FWK_MODULE(GEDGsfElectronCoreProducer);
DEFINE_FWK_MODULE(GEDGsfElectronProducer);
DEFINE_FWK_MODULE(GEDGsfElectronFinalizer);


#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "RecoParticleFlow/PFProducer/plugins/PFProducer.h"
#include "RecoParticleFlow/PFProducer/plugins/PFElectronTranslator.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(PFProducer);
DEFINE_ANOTHER_FWK_MODULE(PFElectronTranslator);

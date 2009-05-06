
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "RecoParticleFlow/PFProducer/plugins/PFProducer.h"
#include "RecoParticleFlow/PFProducer/plugins/PFElectronTranslator.h"
#include "RecoParticleFlow/PFProducer/plugins/PFSimParticleProducer.h"
#include "RecoParticleFlow/PFProducer/plugins/PFBlockProducer.h"
#include "RecoParticleFlow/PFProducer/plugins/TauHadronDecayFilter.h"
#include "RecoParticleFlow/PFProducer/plugins/EFilter.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(PFProducer);
DEFINE_ANOTHER_FWK_MODULE(PFElectronTranslator);
DEFINE_ANOTHER_FWK_MODULE(PFBlockProducer);
DEFINE_ANOTHER_FWK_MODULE(PFSimParticleProducer);
DEFINE_ANOTHER_FWK_MODULE(TauHadronDecayFilter);
DEFINE_ANOTHER_FWK_MODULE(EFilter);

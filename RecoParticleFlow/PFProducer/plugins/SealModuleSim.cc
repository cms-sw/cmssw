
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "RecoParticleFlow/PFProducer/plugins/PFSimParticleProducer.h"
#include "RecoParticleFlow/PFProducer/plugins/TauHadronDecayFilter.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(PFSimParticleProducer);
DEFINE_ANOTHER_FWK_MODULE(TauHadronDecayFilter);

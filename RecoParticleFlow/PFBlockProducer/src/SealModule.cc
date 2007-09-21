// #include "PluginManager/ModuleDef.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFBlockProducer/interface/PFSimParticleProducer.h"
#include "RecoParticleFlow/PFBlockProducer/interface/PFBlockProducer.h"
#include "RecoParticleFlow/PFBlockProducer/interface/TauHadronDecayFilter.h"
#include "RecoParticleFlow/PFBlockProducer/interface/EFilter.h"
// #include "RecoParticleFlow/PFBlockProducer/interface/PFJetCandidateCreator.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(PFBlockProducer);
DEFINE_ANOTHER_FWK_MODULE(PFSimParticleProducer);
DEFINE_ANOTHER_FWK_MODULE(TauHadronDecayFilter);
DEFINE_ANOTHER_FWK_MODULE(EFilter);
// DEFINE_ANOTHER_FWK_MODULE(PFJetCandidateCreator);


#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/typelookup.h"

#include "RecoParticleFlow/PFProducer/plugins/PFSimParticleProducer.h"
#include "RecoParticleFlow/PFProducer/plugins/TauHadronDecayFilter.h"



DEFINE_FWK_MODULE(PFSimParticleProducer);
DEFINE_FWK_MODULE(TauHadronDecayFilter);

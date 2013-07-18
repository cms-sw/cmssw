
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/typelookup.h"

#include "RecoParticleFlow/PFProducer/plugins/PFProducer.h"
#include "RecoParticleFlow/PFProducer/plugins/PFEGammaProducer.h"
#include "RecoParticleFlow/PFProducer/plugins/PFConcretePFCandidateProducer.h"
#include "RecoParticleFlow/PFProducer/plugins/PFCandidateChecker.h"
#include "RecoParticleFlow/PFProducer/plugins/PFElectronTranslator.h"
#include "RecoParticleFlow/PFProducer/plugins/PFPhotonTranslator.h"
#include "RecoParticleFlow/PFProducer/plugins/PFBlockProducer.h"
#include "RecoParticleFlow/PFProducer/plugins/EFilter.h"
#include "RecoParticleFlow/PFProducer/plugins/PFLinker.h"
#include "RecoParticleFlow/PFProducer/plugins/PFCandidateFwdPtrProducer.cc"
#include "RecoParticleFlow/PFProducer/plugins/PFCandidateProductFromFwdPtrProducer.cc"

DEFINE_FWK_MODULE(PFProducer);
DEFINE_FWK_MODULE(PFEGammaProducer);
DEFINE_FWK_MODULE(PFConcretePFCandidateProducer);
DEFINE_FWK_MODULE(PFCandidateChecker);
DEFINE_FWK_MODULE(PFElectronTranslator);
DEFINE_FWK_MODULE(PFPhotonTranslator);
DEFINE_FWK_MODULE(PFBlockProducer);
DEFINE_FWK_MODULE(EFilter);
DEFINE_FWK_MODULE(PFLinker);
DEFINE_FWK_MODULE(PFCandidateFwdPtrProducer);
DEFINE_FWK_MODULE(PFCandidateProductFromFwdPtrProducer);

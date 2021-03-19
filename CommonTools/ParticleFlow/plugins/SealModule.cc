#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/ParticleFlow/plugins/PFMET.h"
#include "CommonTools/ParticleFlow/plugins/PFPileUp.h"
#include "CommonTools/ParticleFlow/plugins/PFCandidateFwdPtrCollectionFilter.h"
#include "CommonTools/ParticleFlow/plugins/PFJetFwdPtrProducer.h"
#include "CommonTools/ParticleFlow/plugins/PFTauFwdPtrProducer.h"
#include "CommonTools/ParticleFlow/plugins/PFCandidateFromFwdPtrProducer.h"
#include "CommonTools/ParticleFlow/plugins/DeltaBetaWeights.h"

DEFINE_FWK_MODULE(PFMET);
DEFINE_FWK_MODULE(PFPileUp);

DEFINE_FWK_MODULE(PFCandidateFwdPtrCollectionStringFilter);
DEFINE_FWK_MODULE(PFCandidateFwdPtrCollectionPdgIdFilter);
DEFINE_FWK_MODULE(PFJetFwdPtrProducer);
DEFINE_FWK_MODULE(PFTauFwdPtrProducer);
DEFINE_FWK_MODULE(PFCandidateFromFwdPtrProducer);

typedef edm::ProductFromFwdPtrProducer<reco::PFJet> PFJetFromFwdPtrProducer;
DEFINE_FWK_MODULE(PFJetFromFwdPtrProducer);

DEFINE_FWK_MODULE(DeltaBetaWeights);

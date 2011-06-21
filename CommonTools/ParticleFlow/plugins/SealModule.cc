#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/ParticleFlow/plugins/PFMET.h"
#include "CommonTools/ParticleFlow/plugins/Type1PFMET.h"
#include "CommonTools/ParticleFlow/plugins/PFPileUp.h"
#include "CommonTools/ParticleFlow/plugins/TopProjector.cc"



DEFINE_FWK_MODULE(PFMET);
DEFINE_FWK_MODULE(Type1PFMET);
DEFINE_FWK_MODULE(PFPileUp);
DEFINE_FWK_MODULE(TPPFCandidatesOnPFCandidates);
DEFINE_FWK_MODULE(TPPFCandidatesOnPileUpPFCandidates);
DEFINE_FWK_MODULE(TPIsolatedPFCandidatesOnPFCandidates);
DEFINE_FWK_MODULE(TPPFJetsOnPFCandidates);
DEFINE_FWK_MODULE(TPPFTausOnPFJets);

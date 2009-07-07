#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/PFCandProducer/plugins/PFMET.h"
#include "PhysicsTools/PFCandProducer/plugins/Type1PFMET.h"
#include "PhysicsTools/PFCandProducer/plugins/PFPileUp.h"
#include "PhysicsTools/PFCandProducer/plugins/TopProjector.cc"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(PFMET);
DEFINE_ANOTHER_FWK_MODULE(Type1PFMET);
DEFINE_ANOTHER_FWK_MODULE(PFPileUp);
DEFINE_ANOTHER_FWK_MODULE(TPPFCandidatesOnPFCandidates);
DEFINE_ANOTHER_FWK_MODULE(TPPileUpPFCandidatesOnPFCandidates);
DEFINE_ANOTHER_FWK_MODULE(TPPFCandidatesOnPileUpPFCandidates);
DEFINE_ANOTHER_FWK_MODULE(TPIsolatedPFCandidatesOnPFCandidates);
DEFINE_ANOTHER_FWK_MODULE(TPPFJetsOnPFCandidates);
DEFINE_ANOTHER_FWK_MODULE(TPPFTausOnPFJets);

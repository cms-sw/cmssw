#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/PFCandProducer/plugins/PFMET.h"
#include "PhysicsTools/PFCandProducer/plugins/Type1PFMET.h"
#include "PhysicsTools/PFCandProducer/plugins/PFPileUp.h"
#include "PhysicsTools/PFCandProducer/plugins/TopProjector.cc"



DEFINE_FWK_MODULE(PFMET);
DEFINE_FWK_MODULE(Type1PFMET);
DEFINE_FWK_MODULE(PFPileUp);
DEFINE_FWK_MODULE(TPPFCandidatesOnPFCandidates);
DEFINE_FWK_MODULE(TPPileUpPFCandidatesOnPFCandidates);
DEFINE_FWK_MODULE(TPPFCandidatesOnPileUpPFCandidates);
DEFINE_FWK_MODULE(TPIsolatedPFCandidatesOnPFCandidates);
DEFINE_FWK_MODULE(TPPFJetsOnPFCandidates);
DEFINE_FWK_MODULE(TPPFTausOnPFJets);

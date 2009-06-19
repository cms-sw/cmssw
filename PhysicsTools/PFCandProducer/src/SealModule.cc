#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/PFCandProducer/interface/PFMET.h"
#include "PhysicsTools/PFCandProducer/interface/Type1PFMET.h"
#include "PhysicsTools/PFCandProducer/interface/PFIsolation.h"
#include "PhysicsTools/PFCandProducer/interface/PFPileUp.h"
#include "PhysicsTools/PFCandProducer/src/PFTopProjector.cc"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(PFMET);
DEFINE_ANOTHER_FWK_MODULE(Type1PFMET);
DEFINE_ANOTHER_FWK_MODULE(PFIsolation);
DEFINE_ANOTHER_FWK_MODULE(PFPileUp);
DEFINE_ANOTHER_FWK_MODULE(PFTopProjectorPFCandidatesOnPFCandidates);
DEFINE_ANOTHER_FWK_MODULE(PFTopProjectorPileUpPFCandidatesOnPFCandidates);
DEFINE_ANOTHER_FWK_MODULE(PFTopProjectorIsolatedPFCandidatesOnPFCandidates);
DEFINE_ANOTHER_FWK_MODULE(PFTopProjectorPFJetsOnPFCandidates);
DEFINE_ANOTHER_FWK_MODULE(PFTopProjectorPFTausOnPFJets);

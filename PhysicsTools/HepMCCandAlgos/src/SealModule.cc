#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/HepMCCandAlgos/src/HepMCCandidateProducer.h"
#include "PhysicsTools/HepMCCandAlgos/src/GenParticleCandidateProducer.h"
#include "PhysicsTools/HepMCCandAlgos/src/FastGenParticleCandidateProducer.h"
#include "PhysicsTools/HepMCCandAlgos/src/GenParticleCandidateSelector.h"
#include "PhysicsTools/HepMCCandAlgos/src/HepMCCandidateSelector.h"
#include "PhysicsTools/HepMCCandAlgos/src/ParticleTreeDrawer.h"
#include "PhysicsTools/HepMCCandAlgos/src/MCTruthDeltaRMatcher.h"
#include "PhysicsTools/HepMCCandAlgos/src/MCTruthCompositeMatcher.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( HepMCCandidateProducer );
DEFINE_ANOTHER_FWK_MODULE( GenParticleCandidateProducer );
DEFINE_ANOTHER_FWK_MODULE( FastGenParticleCandidateProducer );
DEFINE_ANOTHER_FWK_MODULE( GenParticleCandidateSelector );
DEFINE_ANOTHER_FWK_MODULE( HepMCCandidateSelector );
DEFINE_ANOTHER_FWK_MODULE( ParticleTreeDrawer );
DEFINE_ANOTHER_FWK_MODULE( MCTruthDeltaRMatcher );
DEFINE_ANOTHER_FWK_MODULE( MCTruthCompositeMatcher );

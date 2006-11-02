#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/HepMCCandAlgos/src/HepMCCandidateProducer.h"
#include "PhysicsTools/HepMCCandAlgos/src/GenParticleCandidateProducer.h"
#include "PhysicsTools/HepMCCandAlgos/src/HepMCCandidateSelector.h"
#include "PhysicsTools/HepMCCandAlgos/src/SetGenParticleMotherReference.h"
#include "PhysicsTools/HepMCCandAlgos/src/ParticleTreeDrawer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( HepMCCandidateProducer );
DEFINE_ANOTHER_FWK_MODULE( GenParticleCandidateProducer );
DEFINE_ANOTHER_FWK_MODULE( HepMCCandidateSelector );
DEFINE_ANOTHER_FWK_MODULE( SetGenParticleMotherReference );
DEFINE_ANOTHER_FWK_MODULE( ParticleTreeDrawer );

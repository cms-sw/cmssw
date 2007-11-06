#include "PhysicsTools/UtilAlgos/interface/AssociationMerger.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

typedef AssociationMerger<reco::GenParticleCollection> GenParticleMatchMerger;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenParticleMatchMerger );


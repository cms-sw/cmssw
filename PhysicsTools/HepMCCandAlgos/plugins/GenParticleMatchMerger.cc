#include "CommonTools/UtilAlgos/interface/CollectionAdder.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

typedef CollectionAdder<reco::GenParticleMatch> GenParticleMatchMerger;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenParticleMatchMerger );



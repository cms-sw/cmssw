//
//

#include "DataFormats/PatCandidates/interface/PFParticle.h"


using namespace pat;


/// constructor from PFParticleType
PFParticle::PFParticle(const edm::RefToBase<reco::PFCandidate>& aPFParticle) : PATObject<reco::PFCandidate>(aPFParticle) {
}


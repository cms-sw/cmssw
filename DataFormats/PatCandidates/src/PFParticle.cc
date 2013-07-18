//
// $Id: PFParticle.cc,v 1.1 2008/07/24 12:43:52 cbern Exp $
//

#include "DataFormats/PatCandidates/interface/PFParticle.h"


using namespace pat;


/// constructor from PFParticleType
PFParticle::PFParticle(const edm::RefToBase<reco::PFCandidate>& aPFParticle) : PATObject<reco::PFCandidate>(aPFParticle) {
}


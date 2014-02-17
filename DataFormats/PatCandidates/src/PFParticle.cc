//
// $Id: PFParticle.cc,v 1.2 2008/11/28 19:02:15 lowette Exp $
//

#include "DataFormats/PatCandidates/interface/PFParticle.h"


using namespace pat;


/// constructor from PFParticleType
PFParticle::PFParticle(const edm::RefToBase<reco::PFCandidate>& aPFParticle) : PATObject<reco::PFCandidate>(aPFParticle) {
}


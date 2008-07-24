//
// $Id: PFParticle.cc,v 1.1 2008/01/15 12:59:32 lowette Exp $
//

#include "DataFormats/PatCandidates/interface/PFParticle.h"


using namespace pat;


/// constructor from PFParticleType
PFParticle::PFParticle(const edm::RefToBase<PFParticleType>& aPFParticle) : PATObject<PFParticleType>(aPFParticle) {
}


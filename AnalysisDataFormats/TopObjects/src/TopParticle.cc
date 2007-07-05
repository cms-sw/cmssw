//
// $Id: TopParticle.cc,v 1.1 2007/06/23 07:09:29 lowette Exp $
//

#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"


/// default constructor
TopParticle::TopParticle() : TopObject<TopParticleType>(TopParticleType(0, TopParticleType::LorentzVector(0, 0, 0, 0), TopParticleType::Point(0,0,0))) {
}


/// constructor from TopParticleType
TopParticle::TopParticle(const TopParticleType & aParticle) : TopObject<TopParticleType>(aParticle) {
}


/// destructor
TopParticle::~TopParticle() {
}

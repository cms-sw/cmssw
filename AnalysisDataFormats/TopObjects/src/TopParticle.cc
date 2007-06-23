//
// Author:  Steven Lowette
// Created: Fri Jun 22 17:40:46 PDT 2007
//
// $Id$
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

//
// $Id: Particle.cc,v 1.1 2008/01/07 11:48:25 lowette Exp $
//

#include "DataFormats/PatCandidates/interface/Particle.h"


using namespace pat;


/// default constructor
Particle::Particle() : PATObject<ParticleType>(ParticleType(0, ParticleType::LorentzVector(0, 0, 0, 0), ParticleType::Point(0,0,0))) {
}


/// constructor from ParticleType
Particle::Particle(const ParticleType & aParticle) : PATObject<ParticleType>(aParticle) {
}


/// destructor
Particle::~Particle() {
}

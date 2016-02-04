//
// $Id: Particle.cc,v 1.2 2008/11/28 19:02:15 lowette Exp $
//

#include "DataFormats/PatCandidates/interface/Particle.h"


using namespace pat;


/// default constructor
Particle::Particle() : PATObject<reco::LeafCandidate>(reco::LeafCandidate(0, reco::LeafCandidate::LorentzVector(0, 0, 0, 0), reco::LeafCandidate::Point(0,0,0))) {
}


/// constructor from reco::LeafCandidate
Particle::Particle(const reco::LeafCandidate & aParticle) : PATObject<reco::LeafCandidate>(aParticle) {
}


/// destructor
Particle::~Particle() {
}

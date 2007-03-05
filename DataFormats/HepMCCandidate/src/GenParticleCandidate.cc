// $Id: GenParticleCandidate.cc,v 1.10 2007/03/01 15:56:30 llista Exp $
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include <iostream>
using namespace reco;

GenParticleCandidate::GenParticleCandidate( Charge q, const LorentzVector & p4, 
					    const Point & vtx, int pdgId, int status, bool integerCharge ) : 
  CompositeRefCandidate( q, p4, vtx, pdgId, status, integerCharge ) {
}

GenParticleCandidate::~GenParticleCandidate() { }

bool GenParticleCandidate::overlap( const Candidate & c ) const {
  return & c == this;
}

GenParticleCandidate * GenParticleCandidate::clone() const {
  return new GenParticleCandidate( * this );
}

void GenParticleCandidate::fixup() const {
  addMothersFromDaughterLinks();
}

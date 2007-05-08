// $Id: GenParticleCandidate.cc,v 1.11 2007/03/05 13:25:49 llista Exp $
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
  size_t n = numberOfDaughters();
  for( size_t i = 0; i < n; ++ i ) {
    daughter( i )->addMother( this );
  }
}

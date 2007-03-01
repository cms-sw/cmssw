// $Id: GenParticleCandidate.cc,v 1.8 2007/02/19 12:59:05 llista Exp $
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include <iostream>
using namespace reco;

GenParticleCandidate::GenParticleCandidate( Charge q, const LorentzVector & p4, 
					    const Point & vtx, int pdgId, int status ) : 
  CompositeRefCandidate( q, p4, vtx, pdgId, status, false ) {
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

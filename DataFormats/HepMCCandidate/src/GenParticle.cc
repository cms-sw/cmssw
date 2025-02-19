// $Id: GenParticle.cc,v 1.2 2007/10/15 13:03:36 llista Exp $
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include <iostream>
using namespace reco;

GenParticle::GenParticle( Charge q, const LorentzVector & p4, 
			  const Point & vtx, int pdgId, int status, bool integerCharge ) : 
  CompositeRefCandidateT<GenParticleRefVector>( q, p4, vtx, pdgId, status, integerCharge ) {
}

GenParticle::GenParticle( Charge q, const PolarLorentzVector & p4, 
			  const Point & vtx, int pdgId, int status, bool integerCharge ) : 
  CompositeRefCandidateT<GenParticleRefVector>( q, p4, vtx, pdgId, status, integerCharge ) {
}

GenParticle::~GenParticle() { }

bool GenParticle::overlap( const Candidate & c ) const {
  return & c == this;
}

GenParticle * GenParticle::clone() const {
  return new GenParticle( * this );
}

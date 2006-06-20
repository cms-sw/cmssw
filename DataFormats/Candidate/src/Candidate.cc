// $Id: Candidate.cc,v 1.1 2006/02/28 10:43:30 llista Exp $
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace reco;

Candidate::~Candidate() { }

Candidate::setup::~setup() { }

void Candidate::setup::setP4( LorentzVector & p ) const { 
  if ( modifyP4 ) p = p4;
}

void Candidate::setup::setCharge( Charge & q ) const { 
  if ( modifyCharge ) q = charge; 
}

void Candidate::setup::setVertex( Point & v ) const { 
  if ( modifyVertex ) v = vertex; 
}


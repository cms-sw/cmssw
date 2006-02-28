// $Id: Candidate.cc,v 1.5 2006/02/21 10:37:32 llista Exp $
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

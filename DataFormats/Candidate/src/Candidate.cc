// $Id: Candidate.cc,v 1.6 2006/12/07 18:35:49 llista Exp $
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace reco;

Candidate::~Candidate() { }

const CandidateBaseRef & Candidate::masterClone() const {
  throw cms::Exception("Invalid Reference") 
    << "this Candidate has no master clone reference."
    << "Can't call masterClone() method.\n";
}

bool Candidate::hasMasterClone() const { 
  return false;
}

void Candidate::fixup() const {
  for( size_t i = 0; i < numberOfDaughters(); ++ i ) {
    daughter( i )->setMother( this );
  }
}

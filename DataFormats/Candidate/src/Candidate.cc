// $Id: Candidate.cc,v 1.8 2007/02/19 12:59:04 llista Exp $
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

void Candidate::addMothersFromDaughterLinks() const {
  for( size_t i = 0; i < numberOfDaughters(); ++ i ) {
    daughter( i )->addMother( this );
  }
}

// $Id: Candidate.cc,v 1.7 2007/01/19 16:11:48 llista Exp $
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

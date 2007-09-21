// $Id: Candidate.cc,v 1.11 2007/09/14 09:53:43 llista Exp $
#include "DataFormats/Candidate/interface/Candidate.h"
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

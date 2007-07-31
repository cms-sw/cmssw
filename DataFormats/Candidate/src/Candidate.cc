// $Id: Candidate.cc,v 1.9 2007/05/08 13:11:17 llista Exp $
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

// $Id: Candidate.cc,v 1.5 2006/12/07 11:50:34 llista Exp $
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

const Candidate * Candidate::mother() const{
  return 0;
}

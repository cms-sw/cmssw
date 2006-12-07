#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace reco;

ShallowCloneCandidate::~ShallowCloneCandidate() { 
}

ShallowCloneCandidate * ShallowCloneCandidate::clone() const { 
  return new ShallowCloneCandidate( *this ); 
}

ShallowCloneCandidate::const_iterator ShallowCloneCandidate::begin() const { 
  return masterClone_->begin(); 
}

ShallowCloneCandidate::const_iterator ShallowCloneCandidate::end() const { 
  return masterClone_->end(); 
}

ShallowCloneCandidate::iterator ShallowCloneCandidate::begin() { 
  throw cms::Exception("Invalid Reference") << "can't have non-const access to master clone";      
}

ShallowCloneCandidate::iterator ShallowCloneCandidate::end() { 
  throw cms::Exception("Invalid Reference") << "can't have non-const access to master clone";      
}

int ShallowCloneCandidate::numberOfDaughters() const { 
  return masterClone_->numberOfDaughters(); 
}

const Candidate & ShallowCloneCandidate::daughter( size_type i ) const { 
  return masterClone_->daughter( i ); 
}

Candidate & ShallowCloneCandidate::daughter( size_type i ) { 
  throw cms::Exception("Invalid Reference") << "can't have non-const access to master clone";      
}

bool ShallowCloneCandidate::hasMasterClone() const {
  return true;
}

const CandidateBaseRef & ShallowCloneCandidate::masterClone() const {
  return masterClone_;
}


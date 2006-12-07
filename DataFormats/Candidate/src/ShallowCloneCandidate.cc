#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
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

Candidate::iterator ShallowCloneCandidate::begin() { return iterator( new iterator_imp_specific ); }

Candidate::iterator ShallowCloneCandidate::end() { return iterator( new iterator_imp_specific ); }

size_t ShallowCloneCandidate::numberOfDaughters() const { 
  return masterClone_->numberOfDaughters(); 
}

const Candidate * ShallowCloneCandidate::daughter( size_type i ) const { 
  return masterClone_->daughter( i ); 
}

Candidate * ShallowCloneCandidate::daughter( size_type i ) { 
  return 0;
}

bool ShallowCloneCandidate::hasMasterClone() const {
  return true;
}

const CandidateBaseRef & ShallowCloneCandidate::masterClone() const {
  return masterClone_;
}


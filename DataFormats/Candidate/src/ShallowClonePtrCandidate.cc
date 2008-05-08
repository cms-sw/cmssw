#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"
using namespace reco;

ShallowClonePtrCandidate::~ShallowClonePtrCandidate() { 
}

ShallowClonePtrCandidate * ShallowClonePtrCandidate::clone() const { 
  return new ShallowClonePtrCandidate( *this ); 
}

ShallowClonePtrCandidate::const_iterator ShallowClonePtrCandidate::begin() const { 
  return masterClone_->begin(); 
}

ShallowClonePtrCandidate::const_iterator ShallowClonePtrCandidate::end() const { 
  return masterClone_->end(); 
}

Candidate::iterator ShallowClonePtrCandidate::begin() { 
  return iterator( new iterator_imp_specific ); 
}

Candidate::iterator ShallowClonePtrCandidate::end() { 
  return iterator( new iterator_imp_specific ); 
}

size_t ShallowClonePtrCandidate::numberOfDaughters() const { 
  return masterClone_->numberOfDaughters(); 
}

const Candidate * ShallowClonePtrCandidate::daughter( size_type i ) const { 
  return masterClone_->daughter( i ); 
}

Candidate * ShallowClonePtrCandidate::daughter( size_type i ) { 
  return 0;
}

bool ShallowClonePtrCandidate::hasMasterClonePtr() const {
  return true;
}

const CandidatePtr & ShallowClonePtrCandidate::masterClonePtr() const {
  return masterClone_;
}

void ShallowClonePtrCandidate::fixup() const {
}

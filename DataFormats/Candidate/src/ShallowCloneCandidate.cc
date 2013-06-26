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

Candidate::iterator ShallowCloneCandidate::begin() { 
  return iterator( new iterator_imp_specific ); 
}

Candidate::iterator ShallowCloneCandidate::end() { 
  return iterator( new iterator_imp_specific ); 
}

size_t ShallowCloneCandidate::numberOfDaughters() const { 
  return masterClone_->numberOfDaughters(); 
}

size_t ShallowCloneCandidate::numberOfMothers() const { 
  return masterClone_->numberOfMothers(); 
}

const Candidate * ShallowCloneCandidate::daughter( size_type i ) const { 
  return masterClone_->daughter( i ); 
}

const Candidate * ShallowCloneCandidate::mother( size_type i ) const { 
  return masterClone_->mother( i ); 
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

bool ShallowCloneCandidate::isElectron() const { 
  return masterClone_->isElectron(); 
}

bool ShallowCloneCandidate::isMuon() const { 
  return masterClone_->isMuon(); 
}

bool ShallowCloneCandidate::isGlobalMuon() const { 
  return masterClone_->isGlobalMuon(); 
}

bool ShallowCloneCandidate::isStandAloneMuon() const { 
  return masterClone_->isStandAloneMuon(); 
}

bool ShallowCloneCandidate::isTrackerMuon() const { 
  return masterClone_->isTrackerMuon(); 
}

bool ShallowCloneCandidate::isCaloMuon() const { 
  return masterClone_->isCaloMuon(); 
}

bool ShallowCloneCandidate::isPhoton() const { 
  return masterClone_->isPhoton(); 
}

bool ShallowCloneCandidate::isConvertedPhoton() const { 
  return masterClone_->isConvertedPhoton(); 
}

bool ShallowCloneCandidate::isJet() const { 
  return masterClone_->isJet(); 
}

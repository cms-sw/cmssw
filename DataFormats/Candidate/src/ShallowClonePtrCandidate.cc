#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"
using namespace reco;

ShallowClonePtrCandidate::~ShallowClonePtrCandidate() { 
}

ShallowClonePtrCandidate * ShallowClonePtrCandidate::clone() const { 
  return new ShallowClonePtrCandidate( *this ); 
}

size_t ShallowClonePtrCandidate::numberOfDaughters() const { 
  return masterClone_->numberOfDaughters(); 
}

size_t ShallowClonePtrCandidate::numberOfMothers() const { 
  return masterClone_->numberOfMothers(); 
}

const Candidate * ShallowClonePtrCandidate::daughter( size_type i ) const { 
  return masterClone_->daughter( i ); 
}

const Candidate * ShallowClonePtrCandidate::mother( size_type i ) const { 
  return masterClone_->mother( i ); 
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

bool ShallowClonePtrCandidate::isElectron() const { 
  return masterClone_->isElectron(); 
}

bool ShallowClonePtrCandidate::isMuon() const { 
  return masterClone_->isMuon(); 
}

bool ShallowClonePtrCandidate::isGlobalMuon() const { 
  return masterClone_->isGlobalMuon(); 
}

bool ShallowClonePtrCandidate::isStandAloneMuon() const { 
  return masterClone_->isStandAloneMuon(); 
}

bool ShallowClonePtrCandidate::isTrackerMuon() const { 
  return masterClone_->isTrackerMuon(); 
}

bool ShallowClonePtrCandidate::isCaloMuon() const { 
  return masterClone_->isCaloMuon(); 
}

bool ShallowClonePtrCandidate::isPhoton() const { 
  return masterClone_->isPhoton(); 
}

bool ShallowClonePtrCandidate::isConvertedPhoton() const { 
  return masterClone_->isConvertedPhoton(); 
}

bool ShallowClonePtrCandidate::isJet() const { 
  return masterClone_->isJet(); 
}

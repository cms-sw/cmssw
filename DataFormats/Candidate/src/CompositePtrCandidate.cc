#include "DataFormats/Candidate/interface/CompositePtrCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

CompositePtrCandidate::~CompositePtrCandidate() { 
}

CompositePtrCandidate * CompositePtrCandidate::clone() const { 
  return new CompositePtrCandidate( * this ); 
}

const Candidate * CompositePtrCandidate::daughter( size_type i ) const { 
  return ( i < numberOfDaughters() ) ? & * dau[ i ] : nullptr; // i >= 0, since i is unsigned
}

const Candidate * CompositePtrCandidate::mother( size_type i ) const { 
  return nullptr;
}

Candidate * CompositePtrCandidate::daughter( size_type i ) { 
  return nullptr;
}

size_t CompositePtrCandidate::numberOfDaughters() const { 
  return dau.size(); 
}

size_t CompositePtrCandidate::numberOfMothers() const { 
  return 0;
}

size_t CompositePtrCandidate::numberOfSourceCandidatePtrs() const { 
  return numberOfDaughters(); 
}

CandidatePtr CompositePtrCandidate::sourceCandidatePtr( size_type i ) const {
  return daughterPtr(i);
}

bool CompositePtrCandidate::overlap( const Candidate & c2 ) const {
  throw cms::Exception( "Error" ) << "can't check overlap internally for CompositePtrCanddate";
}

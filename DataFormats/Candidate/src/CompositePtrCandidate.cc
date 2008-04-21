// $Id: CompositePtrCandidate.cc,v 1.1 2007/10/29 15:49:20 llista Exp $
#include "DataFormats/Candidate/interface/CompositePtrCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

CompositePtrCandidate::~CompositePtrCandidate() { 
}

CompositePtrCandidate * CompositePtrCandidate::clone() const { 
  return new CompositePtrCandidate( * this ); 
}

Candidate::const_iterator CompositePtrCandidate::begin() const { 
  return const_iterator( new const_iterator_imp_specific( dau.begin() ) ); 
}

Candidate::const_iterator CompositePtrCandidate::end() const { 
  return const_iterator( new const_iterator_imp_specific( dau.end() ) ); 
}    

Candidate::iterator CompositePtrCandidate::begin() { 
  return iterator( new iterator_imp_specific ); 
}

Candidate::iterator CompositePtrCandidate::end() { 
  return iterator( new iterator_imp_specific ); 
}    

const Candidate * CompositePtrCandidate::daughter( size_type i ) const { 
  return ( i >= 0 && i < numberOfDaughters() ) ? & * dau[ i ] : 0;
}

const Candidate * CompositePtrCandidate::mother( size_type i ) const { 
  return 0;
}

Candidate * CompositePtrCandidate::daughter( size_type i ) { 
  return 0;
}

size_t CompositePtrCandidate::numberOfDaughters() const { 
  return dau.size(); 
}

size_t CompositePtrCandidate::numberOfMothers() const { 
  return 0;
}

bool CompositePtrCandidate::overlap( const Candidate & c2 ) const {
  throw cms::Exception( "Error" ) << "can't check overlap internally for CompositePtrCanddate";
}

// $Id: CompositeRefCandidate.cc,v 1.9 2010/12/06 20:04:17 wmtan Exp $
#include "DataFormats/Candidate/interface/CompositeRefCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

CompositeRefCandidate::~CompositeRefCandidate() { 
}

CompositeRefCandidate * CompositeRefCandidate::clone() const { 
  return new CompositeRefCandidate( * this ); 
}

Candidate::const_iterator CompositeRefCandidate::begin() const { 
  return const_iterator( new const_iterator_imp_specific( dau.begin() ) ); 
}

Candidate::const_iterator CompositeRefCandidate::end() const { 
  return const_iterator( new const_iterator_imp_specific( dau.end() ) ); 
}    

Candidate::iterator CompositeRefCandidate::begin() { 
  return iterator( new iterator_imp_specific ); 
}

Candidate::iterator CompositeRefCandidate::end() { 
  return iterator( new iterator_imp_specific ); 
}    

const Candidate * CompositeRefCandidate::daughter( size_type i ) const { 
  return ( i < numberOfDaughters() ) ? & * dau[ i ] : 0; // i >= 0, since i is unsigned
}

const Candidate * CompositeRefCandidate::mother( size_type i ) const { 
  return ( i < numberOfMothers() ) ? & * mom[ i ] : 0; // i >= 0, since i is unsigned
}

Candidate * CompositeRefCandidate::daughter( size_type i ) { 
  return 0;
}

size_t CompositeRefCandidate::numberOfDaughters() const { 
  return dau.size(); 
}

size_t CompositeRefCandidate::numberOfMothers() const { 
  return mom.size();
}

bool CompositeRefCandidate::overlap( const Candidate & c2 ) const {
  throw cms::Exception( "Error" ) << "can't check overlap internally for CompositeRefCanddate";
}

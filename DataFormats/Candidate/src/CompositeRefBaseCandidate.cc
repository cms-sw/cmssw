// $Id: CompositeRefBaseCandidate.cc,v 1.7 2007/09/21 14:13:05 llista Exp $
#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

CompositeRefBaseCandidate::~CompositeRefBaseCandidate() { 
}

CompositeRefBaseCandidate * CompositeRefBaseCandidate::clone() const { 
  return new CompositeRefBaseCandidate( * this ); 
}

Candidate::const_iterator CompositeRefBaseCandidate::begin() const { 
  return const_iterator( new const_iterator_imp_specific( dau.begin() ) ); 
}

Candidate::const_iterator CompositeRefBaseCandidate::end() const { 
  return const_iterator( new const_iterator_imp_specific( dau.end() ) ); 
}    

Candidate::iterator CompositeRefBaseCandidate::begin() { 
  return iterator( new iterator_imp_specific ); 
}

Candidate::iterator CompositeRefBaseCandidate::end() { 
  return iterator( new iterator_imp_specific ); 
}    

const Candidate * CompositeRefBaseCandidate::daughter( size_type i ) const { 
  return ( i >= 0 && i < numberOfDaughters() ) ? & * dau[ i ] : 0;
}

const Candidate * CompositeRefBaseCandidate::mother( size_type i ) const { 
 return 0;
}

Candidate * CompositeRefBaseCandidate::daughter( size_type i ) { 
  return 0;
}

size_t CompositeRefBaseCandidate::numberOfDaughters() const { 
  return dau.size(); 
}

size_t CompositeRefBaseCandidate::numberOfMothers() const { 
  return 0;
}

size_t CompositeRefBaseCandidate::numberOfSourceCandidateRefs() const { 
  return dau.size(); 
}

CandidateBaseRef CompositeRefBaseCandidate::sourceCandidateRef( size_type i ) const {
  return daughterRef(i);
}

bool CompositeRefBaseCandidate::overlap( const Candidate & c2 ) const {
  throw cms::Exception( "Error" ) << "can't check overlap internally for CompositeRefBaseCanddate";
}


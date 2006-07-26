// $Id: CompositeRefBaseCandidate.cc,v 1.1 2006/02/28 10:43:30 llista Exp $
#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

CompositeRefBaseCandidate::~CompositeRefBaseCandidate() { 
}

CompositeRefBaseCandidate * CompositeRefBaseCandidate::clone() const { 
  return new CompositeRefBaseCandidate( * this ); 
}

Candidate::const_iterator CompositeRefBaseCandidate::begin() const { 
  return const_iterator( new const_iterator_comp( dau.begin() ) ); 
}

Candidate::const_iterator CompositeRefBaseCandidate::end() const { 
  return const_iterator( new const_iterator_comp( dau.end() ) ); 
}    

Candidate::iterator CompositeRefBaseCandidate::begin() { 
  return iterator( new iterator_comp( dau.begin() ) ); 
}

Candidate::iterator CompositeRefBaseCandidate::end() { 
  return iterator( new iterator_comp( dau.end() ) ); 
}    

const Candidate & CompositeRefBaseCandidate::daughter( size_type i ) const { 
  return * dau[ i ]; 
}

Candidate & CompositeRefBaseCandidate::daughter( size_type i ) { 
  throw cms::Exception( "Error" ) << "can't have non-const access in CompositeRefBaseCanddate";
}

int CompositeRefBaseCandidate::numberOfDaughters() const { 
  return dau.size(); 
}

bool CompositeRefBaseCandidate::overlap( const Candidate & c2 ) const {
  throw cms::Exception( "Error" ) << "can't check overlap internally for CompositeRefBaseCanddate";
}

// $Id: CompositeRefBaseCandidate.cc,v 1.2 2006/12/05 15:53:00 llista Exp $
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

Candidate * CompositeRefBaseCandidate::daughter( size_type i ) { 
  return 0;
}

size_t CompositeRefBaseCandidate::numberOfDaughters() const { 
  return dau.size(); 
}

bool CompositeRefBaseCandidate::overlap( const Candidate & c2 ) const {
  throw cms::Exception( "Error" ) << "can't check overlap internally for CompositeRefBaseCanddate";
}

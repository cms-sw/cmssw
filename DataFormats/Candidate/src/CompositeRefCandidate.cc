// $Id: CompositeRefCandidate.cc,v 1.2 2006/12/05 15:53:00 llista Exp $
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
  return ( i >= 0 && i < numberOfDaughters() ) ? & * dau[ i ] : 0;
}

Candidate * CompositeRefCandidate::daughter( size_type i ) { 
  return 0;
}

size_t CompositeRefCandidate::numberOfDaughters() const { 
  return dau.size(); 
}

bool CompositeRefCandidate::overlap( const Candidate & c2 ) const {
  throw cms::Exception( "Error" ) << "can't check overlap internally for CompositeRefCanddate";
}

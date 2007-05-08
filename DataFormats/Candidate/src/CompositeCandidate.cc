// $Id: CompositeCandidate.cc,v 1.4 2007/02/19 12:59:05 llista Exp $
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

CompositeCandidate::~CompositeCandidate() { }

CompositeCandidate * CompositeCandidate::clone() const { return new CompositeCandidate( * this ); }

Candidate::const_iterator CompositeCandidate::begin() const { return const_iterator( new const_iterator_imp_specific( dau.begin() ) ); }

Candidate::const_iterator CompositeCandidate::end() const { return const_iterator( new const_iterator_imp_specific( dau.end() ) ); }    

Candidate::iterator CompositeCandidate::begin() { return iterator( new iterator_imp_specific( dau.begin() ) ); }

Candidate::iterator CompositeCandidate::end() { return iterator( new iterator_imp_specific( dau.end() ) ); }    

const Candidate * CompositeCandidate::daughter( size_type i ) const { 
  return ( i >= 0 && i < numberOfDaughters() ) ? & dau[ i ] : 0;
}

Candidate * CompositeCandidate::daughter( size_type i ) { 
  return ( i >= 0 && i < numberOfDaughters() ) ? & dau[ i ] : 0;
}

size_t CompositeCandidate::numberOfDaughters() const { return dau.size(); }

bool CompositeCandidate::overlap( const Candidate & c2 ) const {
  throw cms::Exception( "Error" ) << "can't check overlap internally for CompositeCanddate";
}

void CompositeCandidate::fixup() const {
  size_t n = numberOfDaughters();
  for( size_t i = 0; i < n; ++ i ) {
    daughter( i )->addMother( this );
  }
}

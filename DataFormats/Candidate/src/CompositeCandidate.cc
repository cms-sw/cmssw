// $Id: CompositeCandidate.cc,v 1.7 2007/09/21 14:13:05 llista Exp $
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

CompositeCandidate::CompositeCandidate(const Candidate & c) :
  Candidate(c) {
  size_t n = c.numberOfDaughters();
  for(size_t i = 0; i != n; ++i)
    addDaughter(*c.daughter(i));
}

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
  Candidate * d = ( i >= 0 && i < numberOfDaughters() ) ? & dau[ i ] : 0;
  return d;
}

const Candidate * CompositeCandidate::mother( size_type i ) const { 
  return 0;
}

size_t CompositeCandidate::numberOfDaughters() const { return dau.size(); }

size_t CompositeCandidate::numberOfMothers() const { return 0; }

bool CompositeCandidate::overlap( const Candidate & c2 ) const {
  throw cms::Exception( "Error" ) << "can't check overlap internally for CompositeCanddate";
}

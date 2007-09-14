// $Id: CompositeCandidate.cc,v 1.5 2007/05/08 13:11:17 llista Exp $
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
  Candidate * d = ( i >= 0 && i < numberOfDaughters() ) ? & dau[ i ] : 0;
  if ( d != 0 ) d->addMother( this );
  return d;
}

size_t CompositeCandidate::numberOfDaughters() const { return dau.size(); }

bool CompositeCandidate::overlap( const Candidate & c2 ) const {
  throw cms::Exception( "Error" ) << "can't check overlap internally for CompositeCanddate";
}

void CompositeCandidate::doFixupMothers() const {
}

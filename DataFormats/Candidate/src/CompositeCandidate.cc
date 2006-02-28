// $Id: CompositeCandidate.cc,v 1.7 2006/02/21 10:37:32 llista Exp $
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

CompositeCandidate::~CompositeCandidate() { }

CompositeCandidate * CompositeCandidate::clone() const { return new CompositeCandidate( * this ); }

Candidate::const_iterator CompositeCandidate::begin() const { return const_iterator( new const_iterator_comp( dau.begin() ) ); }

Candidate::const_iterator CompositeCandidate::end() const { return const_iterator( new const_iterator_comp( dau.end() ) ); }    

Candidate::iterator CompositeCandidate::begin() { return iterator( new iterator_comp( dau.begin() ) ); }

Candidate::iterator CompositeCandidate::end() { return iterator( new iterator_comp( dau.end() ) ); }    

const Candidate & CompositeCandidate::daughter( size_type i ) const { return dau[ i ]; }

Candidate & CompositeCandidate::daughter( size_type i ) { return dau[ i ]; }

int CompositeCandidate::numberOfDaughters() const { return dau.size(); }

bool CompositeCandidate::overlap( const Candidate & c2 ) const {
  throw cms::Exception( "Error" ) << "can't check overlap internally for CompositeCanddate";
}

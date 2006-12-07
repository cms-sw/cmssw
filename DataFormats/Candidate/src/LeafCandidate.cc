// $Id: LeafCandidate.cc,v 1.4 2006/12/05 15:53:00 llista Exp $
#include "DataFormats/Candidate/interface/LeafCandidate.h"

using namespace reco;

LeafCandidate::~LeafCandidate() { }

LeafCandidate * LeafCandidate::clone() const {
  return new LeafCandidate( * this );
}

Candidate::const_iterator LeafCandidate::begin() const { return const_iterator( new const_iterator_imp_specific ); }

Candidate::const_iterator LeafCandidate::end() const { return  const_iterator( new const_iterator_imp_specific ); }

Candidate::iterator LeafCandidate::begin() { return iterator( new iterator_imp_specific ); }

Candidate::iterator LeafCandidate::end() { return iterator( new iterator_imp_specific ); }

size_t LeafCandidate::numberOfDaughters() const { return 0; }

bool LeafCandidate::overlap( const Candidate & o ) const { 
  return  p4() == o.p4()&&   vertex() == o.vertex() && charge() == o.charge();
}

const Candidate * LeafCandidate::daughter( size_type ) const {
  return 0;
}

Candidate * LeafCandidate::daughter( size_type ) {
  return 0;
}

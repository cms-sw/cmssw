// $Id: LeafCandidate.cc,v 1.9 2006/02/21 10:37:32 llista Exp $
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace reco;

LeafCandidate::~LeafCandidate() { }

LeafCandidate * LeafCandidate::clone() const {
  return new LeafCandidate( * this );
}

Candidate::const_iterator LeafCandidate::begin() const { return const_iterator( new const_iterator_leaf ); }

Candidate::const_iterator LeafCandidate::end() const { return  const_iterator( new const_iterator_leaf ); }

Candidate::iterator LeafCandidate::begin() { return iterator( new iterator_leaf ); }

Candidate::iterator LeafCandidate::end() { return iterator( new iterator_leaf ); }

int LeafCandidate::numberOfDaughters() const { return 0; }

bool LeafCandidate::overlap( const Candidate & ) const { return false; }

const Candidate & LeafCandidate::daughter( size_type ) const {
  throw edm::Exception( edm::errors::LogicError, "Can't access daughters on a leaf Candidate" );
}

Candidate & LeafCandidate::daughter( size_type ) {
  throw edm::Exception( edm::errors::LogicError, "Can't access daughters on a leaf Candidate" );
}

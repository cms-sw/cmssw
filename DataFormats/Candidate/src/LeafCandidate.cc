// $Id: LeafCandidate.cc,v 1.3 2006/08/28 08:07:25 llista Exp $
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

LeafCandidate::~LeafCandidate() { }

LeafCandidate * LeafCandidate::clone() const {
  return new LeafCandidate( * this );
}

Candidate::const_iterator LeafCandidate::begin() const { return const_iterator( new const_iterator_imp_specific ); }

Candidate::const_iterator LeafCandidate::end() const { return  const_iterator( new const_iterator_imp_specific ); }

Candidate::iterator LeafCandidate::begin() { return iterator( new iterator_imp_specific ); }

Candidate::iterator LeafCandidate::end() { return iterator( new iterator_imp_specific ); }

int LeafCandidate::numberOfDaughters() const { return 0; }

bool LeafCandidate::overlap( const Candidate & o ) const { 
  return  p4() == o.p4()&&   vertex() == o.vertex() && charge() == o.charge();
}

const Candidate & LeafCandidate::daughter( size_type ) const {
  throw cms::Exception( "InvalidReference" ) << "Can't access daughters on a leaf Candidate";
}

Candidate & LeafCandidate::daughter( size_type ) {
  throw cms::Exception( "InvalidReference" ) << "Can't access daughters on a leaf Candidate";
}

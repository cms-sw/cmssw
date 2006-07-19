// BasicJet.cc
// Fedor Ratnikov, UMd
// $Id: BasicJet.cc,v 1.6 2006/06/27 23:15:06 fedor Exp $

//Own header file
#include "DataFormats/JetReco/interface/BasicJet.h"

using namespace reco;

BasicJet::BasicJet (const LorentzVector& fP4, const Point& fVertex) 
  : Jet (fP4, fVertex, 0)
{}

BasicJet* BasicJet::clone () const {
  return new BasicJet (*this);
}

bool BasicJet::overlap( const Candidate & ) const {
  return false;
}

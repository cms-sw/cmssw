// BasicJet.cc
// Fedor Ratnikov, UMd
// $Id: BasicJet.cc,v 1.2 2006/12/06 22:43:24 fedor Exp $

#include <sstream>

//Own header file
#include "DataFormats/JetReco/interface/BasicJet.h"

using namespace reco;

BasicJet::BasicJet (const LorentzVector& fP4, const Point& fVertex, const Jet::Constituents& fConstituents) 
  : Jet (fP4, fVertex, fConstituents)
{}

BasicJet* BasicJet::clone () const {
  return new BasicJet (*this);
}

bool BasicJet::overlap( const Candidate & ) const {
  return false;
}

std::string BasicJet::print () const {
  std::ostringstream out;
  out << Jet::print () // generic jet info
      << "    BasicJet specific: None" << std::endl;
  return out.str ();
}

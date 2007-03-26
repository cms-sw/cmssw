// GenericJet.cc
// Fedor Ratnikov, UMd
// $Id: Jet.cc,v 1.3 2006/12/11 12:21:39 fedor Exp $

#include <sstream>

//Own header file
#include "DataFormats/JetReco/interface/GenericJet.h"

using namespace reco;

GenericJet::GenericJet (const LorentzVector& fP4, 
	  const Point& fVertex, 
	  const std::vector<unsigned>& fConstituents)
  :  CompositeRefBaseCandidate (0, fP4, fVertex),
     mConstituents (fConstituents)
{}

int GenericJet::nConstituents () const {
  return mConstituents.size ();
}

std::vector <unsigned>  GenericJet::getJetConstituents () const {
  return mConstituents;
}

std::string GenericJet::print () const {
  std::ostringstream out;
  out << "GenericJet p/px/py/pz/pt: " << p() << '/' << px () << '/' << py() << '/' << pz() << '/' << pt() << std::endl
      << "    eta/phi: " << eta () << '/' << phi () << std::endl
      << "    # of constituents: " << nConstituents () << std::endl;
  out << "    No Constituents details available" << std::endl;
  return out.str ();
}

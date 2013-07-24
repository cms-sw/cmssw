// GenericJet.cc
// Fedor Ratnikov, UMd
// $Id: GenericJet.cc,v 1.3 2007/05/03 21:13:18 fedor Exp $

#include <sstream>

//Own header file
#include "DataFormats/JetReco/interface/GenericJet.h"

using namespace reco;

GenericJet::GenericJet (const LorentzVector& fP4, 
			const Point& fVertex, 
			const std::vector<CandidateBaseRef>& fConstituents)
  :  CompositeRefBaseCandidate (0, fP4, fVertex)
{
  for (unsigned i = 0; i < fConstituents.size (); i++) addDaughter (fConstituents [i]);
}

int GenericJet::nConstituents () const {
  return numberOfDaughters();
}

std::string GenericJet::print () const {
  std::ostringstream out;
  out << "GenericJet p/px/py/pz/pt: " << p() << '/' << px () << '/' << py() << '/' << pz() << '/' << pt() << std::endl
      << "    eta/phi: " << eta () << '/' << phi () << std::endl
      << "    # of constituents: " << nConstituents () << std::endl;
  out << "    No Constituents details available for this version" << std::endl;
  return out.str ();
}

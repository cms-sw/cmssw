// Jet.cc
// Fedor Ratnikov, UMd
// $Id: Jet.cc,v 1.1 2006/12/06 22:43:24 fedor Exp $

#include <sstream>

//Own header file
#include "DataFormats/JetReco/interface/Jet.h"

using namespace reco;

Jet::Jet (const LorentzVector& fP4, 
	  const Point& fVertex, 
	  const std::vector<reco::CandidateRef>& fConstituents)
  :  CompositeRefCandidate (0, fP4, fVertex)
{
  for (unsigned i = 0; i < fConstituents.size (); i++) addDaughter (fConstituents [i]);
}  

Jet::Constituents Jet::getGonstituents () const {
  Jet::Constituents result;
  for (unsigned i = 0; i < CompositeRefCandidate::numberOfDaughters(); i++) {
    result.push_back (CompositeRefCandidate::daughterRef (i));
  }
  return result;
}


std::string Jet::print () const {
  std::ostringstream out;
  out << "Jet p/px/py/pz/pt: " << p() << '/' << px () << '/' << py() << '/' << pz() << '/' << pt() << std::endl
      << "    eta/phi: " << eta () << '/' << phi () << std::endl
      << "    # of constituents: " << nConstituents () << std::endl;
  out << "    Constituents:" << std::endl;
  Candidate::const_iterator daugh = begin ();
  int index = 0;
  for (; daugh != end (); daugh++, index++) {
    const Candidate* constituent = &*daugh; // deref
    if (constituent) {
      out << "      #" << index << " p/pt/eta/phi: " 
	  << constituent->p() << '/' << constituent->pt() << '/' << constituent->eta() << '/' << constituent->phi() << std::endl; 
    }
    else {
      out << "      #" << index << " constituent is not available in the event"  << std::endl;
    }
  }
  return out.str ();
}

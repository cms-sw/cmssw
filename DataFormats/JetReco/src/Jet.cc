// Jet.cc
// Fedor Ratnikov, UMd
// $Id: Jet.cc,v 1.1 2006/07/19 21:42:05 fedor Exp $

#include <sstream>

//Own header file
#include "DataFormats/JetReco/interface/Jet.h"

using namespace reco;

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

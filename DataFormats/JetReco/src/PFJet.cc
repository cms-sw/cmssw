// PFJet.cc
// Fedor Ratnikov UMd
// $Id: PFJet.cc,v 1.4 2007/05/19 04:26:35 fedor Exp $
#include <sstream>

#include "FWCore/Utilities/interface/Exception.h"

//Own header file
#include "DataFormats/JetReco/interface/PFJet.h"

using namespace reco;

PFJet::PFJet (const LorentzVector& fP4, const Point& fVertex, 
		  const Specific& fSpecific, 
		  const Jet::Constituents& fConstituents)
  : Jet (fP4, fVertex, fConstituents),
    m_specific (fSpecific)
{}

PFJet::PFJet (const LorentzVector& fP4, 
		  const Specific& fSpecific, 
		  const Jet::Constituents& fConstituents)
  : Jet (fP4, Point(0,0,0), fConstituents),
    m_specific (fSpecific)
{}

const reco::PFCandidate* PFJet::getPFCandidate (const reco::Candidate* fConstituent) {
  if (!fConstituent) return 0;
  const reco::Candidate* base = fConstituent;
  if (fConstituent->hasMasterClone ()) 
    base = fConstituent->masterClone().get();
  if (!base) return 0; // not in the event
  const PFCandidate* candidate = dynamic_cast <const PFCandidate*> (base);
  if (!candidate) {
    throw cms::Exception("Invalid Constituent") << "PFJet constituent is not of PFCandidate type."
						<< "Actual type is " << typeid (*base).name();
  }
  return candidate;
}

const reco::PFCandidate* PFJet::getConstituent (unsigned fIndex) const {
  return getPFCandidate (daughter (fIndex));
}

std::vector <const reco::PFCandidate*> PFJet::getConstituents () const {
  std::vector <const reco::PFCandidate*> result;
  for (unsigned i = 0;  i <  numberOfDaughters (); i++) result.push_back (getConstituent (i));
  return result;
}

PFJet* PFJet::clone () const {
  return new PFJet (*this);
}

bool PFJet::overlap( const Candidate & ) const {
  return false;
}

std::string PFJet::print () const {
  std::ostringstream out;
  out << Jet::print () // generic jet info
      << "    PFJet specific:" << std::endl
      << "      charged/neutral hadrons energy: " << chargedHadronEnergy () << '/' << neutralHadronEnergy () << std::endl
      << "      charged/neutral em energy: " << chargedEmEnergy () << '/' << neutralEmEnergy () << std::endl
      << "      charged muon energy: " << chargedMuEnergy () << '/' << std::endl
      << "      charged/neutral multiplicity: " << chargedMultiplicity () << '/' << neutralMultiplicity () << std::endl;
  out << "      PFCandidate constituents:" << std::endl;
  std::vector <const PFCandidate*> constituents = getConstituents ();
  for (unsigned i = 0; i < constituents.size (); ++i) {
    if (constituents[i]) {
      out << "      # " << i << " " << *(constituents[i]) << std::endl;
      reco::PFBlockRef block = constituents[i]->blockRef();
      if (block.isNonnull()) {
	out << "            " << *block << std::endl;
      }
      else {
	out << "            PFBlock is not available in the event" << std::endl;
      }
    }
    else {
      out << "      #" << i << " PFCandidate is not available in the event"  << std::endl;
    }
  }
  return out.str ();
}

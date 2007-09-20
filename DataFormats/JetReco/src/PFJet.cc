// PFJet.cc
// Fedor Ratnikov UMd
// $Id: PFJet.cc,v 1.4 2007/05/19 04:26:35 fedor Exp $
#include <sstream>

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

//Own header file
#include "DataFormats/JetReco/interface/PFJet.h"

using namespace reco;

PFJet::PFJet (const LorentzVector& fP4, const Point& fVertex, 
		  const Specific& fSpecific, 
		  const Jet::Constituents& fConstituents)
  : Jet (fP4, fVertex, fConstituents),
    m_specific (fSpecific)
{}

PFJet::PFJet (const LorentzVector& fP4, const Point& fVertex, 
	      const Specific& fSpecific)
  : Jet (fP4, fVertex),
    m_specific (fSpecific)
{}

PFJet::PFJet (const LorentzVector& fP4, 
		  const Specific& fSpecific, 
		  const Jet::Constituents& fConstituents)
  : Jet (fP4, Point(0,0,0), fConstituents),
    m_specific (fSpecific)
{}

reco::PFBlockRef PFJet::getPFBlock (const reco::Candidate* fConstituent) {
  if (fConstituent) {
    const PFCandidate* candidate = dynamic_cast <const PFCandidate*> (fConstituent);
    if (candidate) {
      throw cms::Exception("Not implemented") << "PFJet::getPFBlock is not implemented";
    }
    else {
      throw cms::Exception("Invalid Constituent") << "PFJet constituent is not of PFCandidate type";
    }
  }
  return PFBlockRef ();
}

reco::PFBlockRef PFJet::getConstituent (unsigned fIndex) const {
  return getPFBlock (daughter (fIndex));
}

std::vector <PFBlockRef> PFJet::getConstituents () const {
  std::vector <PFBlockRef> result;
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
  out << "      PF Blocks: Nothing specific is implemented" << std::endl;
//   std::vector <PFBlockRef> towers = getConstituents ();
//   for (unsigned i = 0; i < towers.size (); i++) {
//     if (towers[i].get ()) {
//       out << "      #" << i << " " << *(towers[i]) << std::endl;
//     }
//     else {
//       out << "      #" << i << " PF Block is not available in the event"  << std::endl;
//     }
//   }
  return out.str ();
}

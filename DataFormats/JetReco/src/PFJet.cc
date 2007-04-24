// PFJet.cc
// Fedor Ratnikov UMd
// $Id: PFJet.cc,v 1.12 2007/02/22 19:17:35 fedor Exp $
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

PFJet::PFJet (const LorentzVector& fP4, 
		  const Specific& fSpecific, 
		  const Jet::Constituents& fConstituents)
  : Jet (fP4, Point(0,0,0), fConstituents),
    m_specific (fSpecific)
{}

reco::PFBlockRef getPFBlock PFJet::caloTower (const reco::Candidate* fConstituent) {
  if (fConstituent) {
//     const RecoCaloTowerCandidate* towerCandidate = dynamic_cast <const RecoCaloTowerCandidate*> (fConstituent);
//     if (towerCandidate) {
//       return towerCandidate->caloTower ();
//     }
//     else {
//       throw cms::Exception("Invalid Constituent") << "PFJet constituent is not of RecoCandidate type";
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
      << "      charged/neutral hadrons pT: " << chargedHadronPt () << '/' << neutralHadronPt () << std::endl
      << "      charged/neutral em pT: " << chargedEmPt () << '/' << neutralEmPt () << std::endl
      << "      charged/neutral multiplicity: " << chargedMultiplicity () << '/' << neutralMultiplicity () << std::endl;
  out << "      PF Blocks:" << std::endl;
  std::vector <PFBlockRef> towers = getConstituents ();
  for (unsigned i = 0; i < towers.size (); i++) {
    if (towers[i].get ()) {
      out << "      #" << i << " " << *(towers[i]) << std::endl;
    }
    else {
      out << "      #" << i << " PF Block is not available in the event"  << std::endl;
    }
  }
  return out.str ();
}

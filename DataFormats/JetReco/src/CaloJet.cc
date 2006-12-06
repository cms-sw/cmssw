// CaloJet.cc
// Fedor Ratnikov UMd
// $Id: CaloJet.cc,v 1.9 2006/06/27 23:15:06 fedor Exp $
#include <sstream>

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"

//Own header file
#include "DataFormats/JetReco/interface/CaloJet.h"

using namespace reco;

CaloJet::CaloJet (const LorentzVector& fP4, const Point& fVertex, 
		  const Specific& fSpecific, 
		  const std::vector<CaloTowerDetId>& fIndices) 
  : Jet (fP4, fVertex, fIndices.size ()),
    m_towerIdxs (fIndices),
    m_specific (fSpecific)
{}

CaloJet::CaloJet (const LorentzVector& fP4, 
		  const Specific& fSpecific, 
		  const std::vector<CaloTowerDetId>& fIndices) 
  : Jet (fP4, Point(0,0,0), fIndices.size ()),
    m_towerIdxs (fIndices),
    m_specific (fSpecific)
{}

CaloTowerRef CaloJet::getConstituent (unsigned fIndex) const {
  // no direct access, have to iterate for now
  int index (fIndex);
  Candidate::const_iterator daugh = begin ();
  for (; --index >= 0 && daugh != end (); daugh++) {}
  if (daugh != end ()) { // in range
    const Candidate* constituent = &*daugh; // deref
    if (constituent) {
      const RecoCaloTowerCandidate* towerCandidate = dynamic_cast <const RecoCaloTowerCandidate*> (constituent);
      if (towerCandidate) {
	return (towerCandidate->caloTower ());
      }
      else {
	throw cms::Exception("Invalid Constituent") << "CaloJet constituent is not of RecoCaloTowerCandidate type";
      }
    }
  }
  return CaloTowerRef ();
}


std::vector <CaloTowerRef> CaloJet::getConstituents () const {
  std::vector <CaloTowerRef> result;
  for (int i = 0;  i <  numberOfDaughters (); i++) result.push_back (getConstituent (i));
  return result;
}


CaloJet* CaloJet::clone () const {
  return new CaloJet (*this);
}

bool CaloJet::overlap( const Candidate & ) const {
  return false;
}

std::string CaloJet::print () const {
  std::ostringstream out;
  out << Jet::print () // generic jet info
      << "    CaloJet specific:" << std::endl
      << "      energy fractions em/had: " << emEnergyFraction () << '/' << energyFractionHadronic () << std::endl
      << "      em energy in EB/EE/HF: " << emEnergyInEB() << '/' << emEnergyInEE() << '/' << emEnergyInHF() << std::endl
      << "      had energy in HB/HO/HE/HF: " << hadEnergyInHB() << '/' << hadEnergyInHO() << '/' << hadEnergyInHE() << '/' << hadEnergyInHF() << std::endl;
  out << "      Towers:" << std::endl;
  std::vector <CaloTowerRef> towers = getConstituents ();
  for (unsigned i = 0; i < towers.size (); i++) {
    if (towers[i].get ()) {
      out << "      #" << i << " " << *(towers[i]) << std::endl;
    }
    else {
      out << "      #" << index << " tower is not available in the event"  << std::endl;
    }
  }
  return out.str ();
}

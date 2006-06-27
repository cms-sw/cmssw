// CaloJet.cc
// Fedor Ratnikov UMd
// $Id: CaloJet.cc,v 1.8 2006/05/24 00:40:44 fedor Exp $
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

CaloJet* CaloJet::clone () const {
  return new CaloJet (*this);
}

bool CaloJet::overlap( const Candidate & ) const {
  return false;
}

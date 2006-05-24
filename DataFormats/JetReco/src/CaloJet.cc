// CaloJet.cc
// Fedor Ratnikov UMd
// $Id$
//Own header file
#include "DataFormats/JetReco/interface/CaloJet.h"

CaloJet::CaloJet (const LorentzVector& fP4, const Specific& fSpecific, 
		  const std::vector<CaloTowerDetId>& fIndices) 
  : Jet (fP4),
    m_towerIdxs (fIndices),
    m_specific (fSpecific)
{
  setNConstituents (fIndices.size ());
}


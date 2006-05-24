// GenJet.cc
// Fedor Ratnikov, UMd
// $Id: GenJet.cc,v 1.4 2006/05/18 10:31:50 llista Exp $

//Own header file
#include "DataFormats/JetReco/interface/GenJet.h"

GenJet::GenJet (const LorentzVector& fP4, const Specific& fSpecific, 
		  const std::vector<int>& fBarcodes)
  : Jet (fP4),
    m_barcodes (fBarcodes),
    m_specific (fSpecific)
{
  setNConstituents (fBarcodes.size ());
}

// GenJet.cc
// Fedor Ratnikov, UMd
// $Id: GenJet.cc,v 1.5 2006/05/24 00:40:44 fedor Exp $

//Own header file
#include "DataFormats/JetReco/interface/GenJet.h"

using namespace reco;

GenJet::GenJet (const LorentzVector& fP4, const Point& fVertex, 
		const Specific& fSpecific, 
		const std::vector<int>& fBarcodes)
  : Jet (fP4, fVertex, fBarcodes.size ()),
    m_barcodes (fBarcodes),
    m_specific (fSpecific)
{}

GenJet::GenJet (const LorentzVector& fP4, 
		const Specific& fSpecific, 
		const std::vector<int>& fBarcodes)
  : Jet (fP4, Point(0,0,0), fBarcodes.size ()),
    m_barcodes (fBarcodes),
    m_specific (fSpecific)
{}

GenJet* GenJet::clone () const {
  return new GenJet (*this);
}

bool GenJet::overlap( const Candidate & ) const {
  return false;
}

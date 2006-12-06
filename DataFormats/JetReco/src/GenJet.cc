// GenJet.cc
// Fedor Ratnikov, UMd
// $Id: GenJet.cc,v 1.6 2006/06/27 23:15:06 fedor Exp $

#include <sstream>

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

std::string GenJet::print () const {
  std::ostringstream out;
  out << Jet::print () // generic jet info
      << "    GenJet specific:" << std::endl
      << "      em/had/invisible/aux  energies: " 
      << emEnergy() << '/' << hadEnergy()  << '/' << invisibleEnergy() << '/' << auxiliaryEnergy() << std::endl;
  return out.str ();
}

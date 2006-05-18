// GenJet.cc
// Fedor Ratnikov, UMd
// $Id: GenJet.cc,v 1.3 2006/04/27 18:44:03 fedor Exp $

//Own header file
#include "DataFormats/JetReco/interface/GenJet.h"

// Jet four-momentum
double GenJet::px() const {return m_data.mP4.Px();}
double GenJet::py() const {return m_data.mP4.Py();}
double GenJet::pz() const {return m_data.mP4.Pz();}
double GenJet::energy() const {return m_data.mP4.E();}

// Standard quantities derived from the Jet Lorentz vector
double GenJet::p() const {return m_data.mP4.P();}
double GenJet::pt() const {return m_data.mP4.Pt();}
double GenJet::et() const {return m_data.mP4.Et();}
double GenJet::mass() const {return m_data.mP4.M();}
double GenJet::phi() const {return m_data.mP4.Phi();}
double GenJet::eta() const {return m_data.mP4.Eta();}
int GenJet::nConstituents() const {return m_data.numberOfConstituents;}
Jet::LorentzVector GenJet::p4() const { return m_data.mP4; }
Jet::Vector GenJet::momentum() const { return m_data.mP4.Vect(); }

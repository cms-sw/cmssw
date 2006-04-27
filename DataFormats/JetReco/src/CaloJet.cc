// CaloJet.cc
// Initial Version From Fernando Varela Rodriguez
// Revisions: R. Harris, 19-Oct-2005, modified to work with real 
//            CaloTowers from Jeremy Mans.  Commented out energy
//            fractions until we can figure out how to determine 
//            composition of total energy, and the underlying HB, HE, 
//            HF, HO and Ecal.

//Own header file
#include "DataFormats/JetReco/interface/CaloJet.h"

// Jet four-momentum
double CaloJet::px() const {return m_data.mP4.Px();}
double CaloJet::py() const {return m_data.mP4.Py();}
double CaloJet::pz() const {return m_data.mP4.Pz();}
double CaloJet::energy() const {return m_data.mP4.E();}

// Standard quantities derived from the Jet Lorentz vector
double CaloJet::p() const {return m_data.mP4.P();}
double CaloJet::pt() const {return m_data.mP4.Pt();}
double CaloJet::et() const {return m_data.mP4.Et();}
double CaloJet::mass() const {return m_data.mP4.M();}
double CaloJet::phi() const {return m_data.mP4.Phi();}
double CaloJet::eta() const {return m_data.mP4.Eta();}
int CaloJet::nConstituents() const {return m_data.numberOfConstituents;}


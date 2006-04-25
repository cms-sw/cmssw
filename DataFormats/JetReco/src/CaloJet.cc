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
double CaloJet::px() const {return m_data.px;}
double CaloJet::py() const {return m_data.py;}
double CaloJet::pz() const {return m_data.pz;}
double CaloJet::energy() const {return m_data.e;}

// Standard quantities derived from the Jet Lorentz vector
double CaloJet::p() const {return m_data.p;}
double CaloJet::pt() const {return m_data.pt;}
double CaloJet::et() const {return m_data.et;}
double CaloJet::mass() const {return m_data.m;}
double CaloJet::phi() const {return m_data.phi;}
double CaloJet::eta() const {return m_data.eta;}
int CaloJet::nConstituents() const {return m_data.numberOfConstituents;}


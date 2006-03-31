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
double CaloJet::getPx() const {return m_data.px;}
double CaloJet::getPy() const {return m_data.py;}
double CaloJet::getPz() const {return m_data.pz;}
double CaloJet::getE() const {return m_data.e;}

// Standard quantities derived from the Jet Lorentz vector
double CaloJet::getP() const {return m_data.p;}
double CaloJet::getPt() const {return m_data.pt;}
double CaloJet::getEt() const {return m_data.et;}
double CaloJet::getM() const {return m_data.m;}
double CaloJet::getPhi() const {return m_data.phi;}
double CaloJet::getEta() const {return m_data.eta;}
double CaloJet::getY() const {return m_data.y;}
int CaloJet::getNConstituents() const {return m_data.numberOfConstituents;}


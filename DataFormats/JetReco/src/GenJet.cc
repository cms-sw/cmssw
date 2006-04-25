// GenJet.cc
// Fedor Ratnikov, UMd
// $Id: GenJet.cc,v 1.1 2006/04/05 00:18:42 fedor Exp $

//Own header file
#include "DataFormats/JetReco/interface/GenJet.h"

// Jet four-momentum
double GenJet::px() const {return m_data.px;}
double GenJet::py() const {return m_data.py;}
double GenJet::pz() const {return m_data.pz;}
double GenJet::energy() const {return m_data.e;}

// Standard quantities derived from the Jet Lorentz vector
double GenJet::p() const {return m_data.p;}
double GenJet::pt() const {return m_data.pt;}
double GenJet::et() const {return m_data.et;}
double GenJet::mass() const {return m_data.m;}
double GenJet::phi() const {return m_data.phi;}
double GenJet::eta() const {return m_data.eta;}
int GenJet::nConstituents() const {return m_data.numberOfConstituents;}

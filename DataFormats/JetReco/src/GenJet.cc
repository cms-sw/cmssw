// GenJet.cc
// Fedor Ratnikov, UMd
// $Id: GenJet.h,v 1.6 2006/02/22 19:51:38 fedor Exp $

//Own header file
#include "DataFormats/JetReco/interface/GenJet.h"

// Jet four-momentum
double GenJet::getPx() const {return m_data.px;}
double GenJet::getPy() const {return m_data.py;}
double GenJet::getPz() const {return m_data.pz;}
double GenJet::getE() const {return m_data.e;}

// Standard quantities derived from the Jet Lorentz vector
double GenJet::getP() const {return m_data.p;}
double GenJet::getPt() const {return m_data.pt;}
double GenJet::getEt() const {return m_data.et;}
double GenJet::getM() const {return m_data.m;}
double GenJet::getPhi() const {return m_data.phi;}
double GenJet::getEta() const {return m_data.eta;}
double GenJet::getY() const {return m_data.y;}
int GenJet::getNConstituents() const {return m_data.numberOfConstituents;}




#include "DataFormats/L1Trigger/interface/L1PFTau.h"

using std::ostream;
using std::endl;
using std::hex;
using std::dec;

typedef std::vector<l1t::L1PFTau> L1PFTauCollection;

// default constructor
l1t::L1PFTau::L1PFTau() : m_data(0), m_relativeIsolation(100), m_tauType(12){ }


// destructor
l1t::L1PFTau::~L1PFTau() { }


// print to stream

ostream& l1t::operator<<(ostream& os, const l1t::L1PFTau& tau) {

  os << "L1PFTau:";
  os << " Reco -> ET = " <<tau.p4().Pt();
          os <<" Eta = " <<tau.p4().Eta();
	  os <<" Phi = " <<tau.p4().Phi() << std::endl;

  os << " Tau Decay Mode = "<< tau.tauType() << std::endl;
  os << " Et = "      << tau.et();
  os << " towerEta = "<< tau.towerEta();
  os << " towerPhi = "<< tau.towerPhi()<<std::endl;

  os << " ecalEnergy = "<< tau.ecalEnergy();
  os << " hcalEnergy = "<< tau.hcalEnergy();
  os << " caloEnergy = "<< tau.caloEnergy()<<std::endl;

  os << " EoH = "<<tau.EoH();
  os  <<" HoE = "<<tau.HoE()<<std::endl;

  os  <<" rawIso = "<<tau.rawIso()<<std::endl;
  os  <<" relIso = "<<tau.relIso()<<std::endl;

  os << " raw = "<<tau.raw()<<std::endl;

  return os;
}


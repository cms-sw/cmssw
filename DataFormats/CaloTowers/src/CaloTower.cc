#include "DataFormats/CaloTowers/interface/CaloTower.h"

CaloTower::CaloTower() {
  emE_=0;
  hadE_=0;
  outerE_=0;
  emLvl1_=0;
  hadLvl1_=0;
}

CaloTower::CaloTower(const CaloTowerDetId& id,
		     double emE, double hadE, double outerE,
		     int ecal_tp, int hcal_tp,
		     const PolarLorentzVector p4,
         GlobalPoint emPos, GlobalPoint hadPos) : 
  LeafCandidate(0, p4, Point(0,0,0)),  
  id_(id),
  emE_(emE), hadE_(hadE), outerE_(outerE),
  emLvl1_(ecal_tp), hadLvl1_(hcal_tp),
  emPosition_(emPos), hadPosition_(hadPos)  {}
  

CaloTower::CaloTower(const CaloTowerDetId& id,
		     double emE, double hadE, double outerE,
		     int ecal_tp, int hcal_tp,
		     const LorentzVector p4,
         GlobalPoint emPos, GlobalPoint hadPos) : 
  LeafCandidate(0, p4, Point(0,0,0)),  
  id_(id),
  emE_(emE), hadE_(hadE), outerE_(outerE),
  emLvl1_(ecal_tp), hadLvl1_(hcal_tp),
  emPosition_(emPos), hadPosition_(hadPos)  {}


void CaloTower::addConstituents( const std::vector<DetId>& ids ) {
  constituents_.reserve(constituents_.size()+ids.size());
  constituents_.insert(constituents_.end(),ids.begin(),ids.end());
}

int CaloTower::numCrystals() {
  if (id_.ietaAbs()>27) return 0;
  
  int nC = 0;
  std::vector<DetId>::iterator it = constituents_.begin();
  for (; it!=constituents_.end(); ++it) {
    if (it->det()==DetId::Ecal) ++nC;
  }

  return nC;
}

std::ostream& operator<<(std::ostream& s, const CaloTower& ct) {
  return s << ct.id() << ":" << ct.et() << " GeV ET (EM=" << ct.emEt() <<
    " HAD=" << ct.hadEt() << " OUTER=" << ct.outerEt() << ") (" << ct.eta() << "," << ct.phi() << ")";    
}

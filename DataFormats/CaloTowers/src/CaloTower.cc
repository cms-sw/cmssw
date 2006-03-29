#include "DataFormats/CaloTowers/interface/CaloTower.h"

CaloTower::CaloTower() {
  emEt_=0;
  hadEt_=0;
  outerEt_=0;
  emLvl1_=0;
  hadLvl1_=0;
}

CaloTower::CaloTower(const CaloTowerDetId& id, const Vector& vec, 
	    double emEt, double hadEt, double outerEt,
		     int ecal_tp, int hcal_tp) :
  id_(id),momentum_(vec),
  emEt_(emEt),hadEt_(hadEt),outerEt_(outerEt),
  emLvl1_(ecal_tp),hadLvl1_(hcal_tp) {
}

void CaloTower::addConstituents( const std::vector<DetId>& ids ) {
  constituents_.reserve(constituents_.size()+ids.size());
  constituents_.insert(constituents_.end(),ids.begin(),ids.end());
}

std::ostream& operator<<(std::ostream& s, const CaloTower& ct) {
  return s << ct.id() << ":" << ct.et() << " GeV ET (EM=" << ct.emEt() <<
    " HAD=" << ct.hadEt() << " OUTER=" << ct.outerEt() << ") (" << ct.eta() << "," << ct.phi() << ")";    
}

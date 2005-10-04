#include "DataFormats/CaloTowers/interface/CaloTower.h"

CaloTower::CaloTower() {
  eT=0;
  eT_em=0;
  eT_had=0;
  eT_outer=0;
  eta=0;
  phi=0;
}

CaloTower::CaloTower(const CaloTowerDetId& id) : id_(id) {
  eT=0;
  eT_em=0;
  eT_had=0;
  eT_outer=0;
  eta=0;
  phi=0;
}

std::ostream& operator<<(std::ostream& s, const CaloTower& ct) {
  return s << ct.id_ << ":" << ct.eT << " GeV (EM=" << ct.eT_em <<
    " HAD=" << ct.eT_had << " OUTER=" << ct.eT_outer << ") (" << ct.eta << "," << ct.phi << ")";    
}

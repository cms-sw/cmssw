
#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

namespace l1t::io_v1 {
  CaloTower::CaloTower(const LorentzVector& p4,
                       double etEm,
                       double etHad,
                       int pt,
                       int eta,
                       int phi,
                       int qual,
                       int hwEtEm,
                       int hwEtHad,
                       int hwEtRatio)
      : L1Candidate(p4, pt, eta, phi, qual),
        etEm_(etEm),
        etHad_(etHad),
        hwEtEm_(hwEtEm),
        hwEtHad_(hwEtHad),
        hwEtRatio_(hwEtRatio) {}

  CaloTower::~CaloTower() {}

  void CaloTower::setEtEm(double et) { etEm_ = et; }

  void CaloTower::setEtHad(double et) { etHad_ = et; }

  void CaloTower::setHwEtEm(int et) { hwEtEm_ = et; }

  void CaloTower::setHwEtHad(int et) { hwEtHad_ = et; }

  void CaloTower::setHwEtRatio(int ratio) { hwEtRatio_ = ratio; }

  double CaloTower::etEm() const { return etEm_; }

  double CaloTower::etHad() const { return etHad_; }

  int CaloTower::hwEtEm() const { return hwEtEm_; }

  int CaloTower::hwEtHad() const { return hwEtHad_; }

  int CaloTower::hwEtRatio() const { return hwEtRatio_; }
}  // namespace l1t::io_v1

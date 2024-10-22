
#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

l1t::CaloTower::CaloTower(const LorentzVector& p4,
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

l1t::CaloTower::~CaloTower() {}

void l1t::CaloTower::setEtEm(double et) { etEm_ = et; }

void l1t::CaloTower::setEtHad(double et) { etHad_ = et; }

void l1t::CaloTower::setHwEtEm(int et) { hwEtEm_ = et; }

void l1t::CaloTower::setHwEtHad(int et) { hwEtHad_ = et; }

void l1t::CaloTower::setHwEtRatio(int ratio) { hwEtRatio_ = ratio; }

double l1t::CaloTower::etEm() const { return etEm_; }

double l1t::CaloTower::etHad() const { return etHad_; }

int l1t::CaloTower::hwEtEm() const { return hwEtEm_; }

int l1t::CaloTower::hwEtHad() const { return hwEtHad_; }

int l1t::CaloTower::hwEtRatio() const { return hwEtRatio_; }

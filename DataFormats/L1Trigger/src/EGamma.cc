
#include "DataFormats/L1Trigger/interface/EGamma.h"
using namespace l1t;

void EGamma::clear_extended() {
  towerIEta_ = 0;
  towerIPhi_ = 0;
  rawEt_ = 0;
  isoEt_ = 0;
  footprintEt_ = 0;
  nTT_ = 0;
  shape_ = 0;
  towerHoE_ = 0;
}

EGamma::EGamma(const LorentzVector& p4, int pt, int eta, int phi, int qual, int iso)
    : L1Candidate(p4, pt, eta, phi, qual, iso) {
  clear_extended();
}

EGamma::EGamma(const PolarLorentzVector& p4, int pt, int eta, int phi, int qual, int iso)
    : L1Candidate(p4, pt, eta, phi, qual, iso) {
  clear_extended();
}

EGamma::~EGamma() {}

void EGamma::setTowerIEta(short int ieta) { towerIEta_ = ieta; }

void EGamma::setTowerIPhi(short int iphi) { towerIPhi_ = iphi; }

void EGamma::setRawEt(short int et) { rawEt_ = et; }

void EGamma::setIsoEt(short int et) { isoEt_ = et; }

void EGamma::setFootprintEt(short int et) { footprintEt_ = et; }

void EGamma::setNTT(short int ntt) { nTT_ = ntt; }

void EGamma::setShape(short int s) { shape_ = s; }

void EGamma::setTowerHoE(short int HoE) { towerHoE_ = HoE; }

short int EGamma::towerIEta() const { return towerIEta_; }

short int EGamma::towerIPhi() const { return towerIPhi_; }

short int EGamma::rawEt() const { return rawEt_; }

short int EGamma::isoEt() const { return isoEt_; }

short int EGamma::footprintEt() const { return footprintEt_; }

short int EGamma::nTT() const { return nTT_; }

short int EGamma::shape() const { return shape_; }

short int EGamma::towerHoE() const { return towerHoE_; }

bool EGamma::operator==(const l1t::EGamma& rhs) const {
  return l1t::L1Candidate::operator==(static_cast<const l1t::L1Candidate&>(rhs)) && towerIEta_ == rhs.towerIEta() &&
         towerIPhi_ == rhs.towerIPhi() && rawEt_ == rhs.rawEt() && isoEt_ == rhs.isoEt() &&
         footprintEt_ == rhs.footprintEt() && nTT_ == rhs.nTT() && shape_ == rhs.shape() && towerHoE_ == rhs.towerHoE();
}

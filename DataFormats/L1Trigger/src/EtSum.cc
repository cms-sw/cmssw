#include "DataFormats/L1Trigger/interface/EtSum.h"

namespace l1t::io_v1 {
  EtSum::EtSum(const LorentzVector& p4, EtSumType type, int pt, int eta, int phi, int qual)
      : L1Candidate(p4, pt, eta, phi, qual, 0), type_(type) {}

  EtSum::EtSum(const PolarLorentzVector& p4, EtSumType type, int pt, int eta, int phi, int qual)
      : L1Candidate(p4, pt, eta, phi, qual, 0), type_(type) {}

  EtSum::~EtSum() {}

  void EtSum::setType(EtSumType type) { type_ = type; }

  EtSum::EtSumType l1t::EtSum::getType() const { return type_; }

  bool EtSum::operator==(const l1t::EtSum& rhs) const {
    return l1t::L1Candidate::operator==(static_cast<const l1t::L1Candidate&>(rhs)) && type_ == rhs.getType();
  }
}  // namespace l1t::io_v1

#include "DataFormats/L1Trigger/interface/MuonShower.h"

l1t::MuonShower::MuonShower(bool oneNominalInTime, bool oneNominalOutOfTime, bool twoLooseInTime, bool twoLooseOutOfTime)
    : L1Candidate(math::PtEtaPhiMLorentzVector{0., 0., 0., 0.}, 0., 0., 0., 0, 0),
      isOneNominalInTime_(oneNominalInTime),
      isOneNominalOutOfTime_(oneNominalOutOfTime),
      isTwoLooseInTime_(twoLooseInTime),
      isTwoLooseOutOfTime_(twoLooseOutOfTime) {}

l1t::MuonShower::~MuonShower() {}

bool l1t::MuonShower::isValid() const {
  return isOneNominalInTime_ or isTwoLooseInTime_ or isOneNominalOutOfTime_ or isTwoLooseOutOfTime_;
}

bool l1t::MuonShower::operator==(const l1t::MuonShower& rhs) const {
  return (isTwoLooseInTime_ == rhs.isTwoLooseInTime() and isOneNominalInTime_ == rhs.isOneNominalInTime() and
          isTwoLooseOutOfTime_ == rhs.isTwoLooseOutOfTime() and isOneNominalOutOfTime_ == rhs.isOneNominalOutOfTime());
}

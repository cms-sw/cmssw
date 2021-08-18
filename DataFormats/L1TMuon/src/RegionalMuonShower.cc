#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"

l1t::RegionalMuonShower::RegionalMuonShower(bool oneNominalInTime,
                                            bool oneNominalOutOfTime,
                                            bool twoLooseInTime,
                                            bool twoLooseOutOfTime)
    : isOneNominalInTime_(oneNominalInTime),
      isOneNominalOutOfTime_(oneNominalOutOfTime),
      isTwoLooseInTime_(twoLooseInTime),
      isTwoLooseOutOfTime_(twoLooseOutOfTime),
      endcap_(0),
      sector_(0),
      link_(0) {}

l1t::RegionalMuonShower::~RegionalMuonShower() {}

bool l1t::RegionalMuonShower::isValid() const {
  return isOneNominalInTime_ or isTwoLooseInTime_ or isOneNominalOutOfTime_ or isTwoLooseOutOfTime_;
}

bool l1t::RegionalMuonShower::operator==(const l1t::RegionalMuonShower& rhs) const {
  return (isTwoLooseInTime_ == rhs.isTwoLooseInTime() and isOneNominalInTime_ == rhs.isOneNominalInTime() and
          isTwoLooseOutOfTime_ == rhs.isTwoLooseOutOfTime() and isOneNominalOutOfTime_ == rhs.isOneNominalOutOfTime());
}

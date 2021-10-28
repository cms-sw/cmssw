#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"

l1t::RegionalMuonShower::RegionalMuonShower(bool oneNominalInTime,
                                            bool oneNominalOutOfTime,
                                            bool twoLooseInTime,
                                            bool twoLooseOutOfTime,
                                            bool oneTightInTime,
                                            bool oneTightOutOfTime)
    : isOneNominalInTime_(oneNominalInTime),
      isOneNominalOutOfTime_(oneNominalOutOfTime),
      isOneTightInTime_(oneTightInTime),
      isOneTightOutOfTime_(oneTightOutOfTime),
      isTwoLooseInTime_(twoLooseInTime),
      isTwoLooseOutOfTime_(twoLooseOutOfTime),
      endcap_(0),
      sector_(0),
      link_(0) {}

l1t::RegionalMuonShower::~RegionalMuonShower() {}

bool l1t::RegionalMuonShower::isValid() const {
  return (isOneNominalInTime_ or isTwoLooseInTime_ or isOneTightInTime_);
}

bool l1t::RegionalMuonShower::operator==(const l1t::RegionalMuonShower& rhs) const {
  return (isTwoLooseInTime_ == rhs.isTwoLooseInTime() and isOneNominalInTime_ == rhs.isOneNominalInTime() and
          isOneTightInTime_ == rhs.isOneTightInTime());
}

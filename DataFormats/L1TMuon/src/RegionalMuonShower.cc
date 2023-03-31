#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"

l1t::RegionalMuonShower::RegionalMuonShower(bool oneNominalInTime,
                                            bool oneNominalOutOfTime,
                                            bool twoLooseInTime,
                                            bool twoLooseOutOfTime,
                                            bool oneLooseInTime,
                                            bool oneTightInTime,
                                            bool oneTightOutOfTime)
    : isOneNominalInTime_(oneNominalInTime),
      isOneNominalOutOfTime_(oneNominalOutOfTime),
      isOneTightInTime_(oneTightInTime),
      isOneTightOutOfTime_(oneTightOutOfTime),
      isTwoLooseInTime_(twoLooseInTime),
      isTwoLooseOutOfTime_(twoLooseOutOfTime),
      isOneLooseInTime_(oneLooseInTime),
      link_(0),
      processor_(0),
      trackFinder_(l1t::tftype::bmtf) {}

l1t::RegionalMuonShower::~RegionalMuonShower() {}

void l1t::RegionalMuonShower::setTFIdentifiers(int processor, tftype trackFinder) {
  trackFinder_ = trackFinder;
  processor_ = processor;

  switch (trackFinder_) {
    case tftype::emtf_pos:
      link_ = processor_ + 36;  // range 36...41
      break;
    case tftype::omtf_pos:
      link_ = processor_ + 42;  // range 42...47
      break;
    case tftype::bmtf:
      link_ = processor_ + 48;  // range 48...59
      break;
    case tftype::omtf_neg:
      link_ = processor_ + 60;  // range 60...65
      break;
    case tftype::emtf_neg:
      link_ = processor_ + 66;  // range 66...71
  }
}

bool l1t::RegionalMuonShower::isValid() const {
  return (isOneNominalInTime_ or isTwoLooseInTime_ or isOneTightInTime_ or isOneLooseInTime_);
}

bool l1t::RegionalMuonShower::operator==(const l1t::RegionalMuonShower& rhs) const {
  return (isTwoLooseInTime_ == rhs.isTwoLooseInTime() and isOneNominalInTime_ == rhs.isOneNominalInTime() and
          isOneTightInTime_ == rhs.isOneTightInTime() and isOneLooseInTime_ == rhs.isOneLooseInTime());
}

#include "DataFormats/L1Trigger/interface/MuonShower.h"

l1t::MuonShower::MuonShower(bool oneNominalInTime,
                            bool oneNominalOutOfTime,
                            bool twoLooseInTime,
                            bool twoLooseOutOfTime,
                            bool oneTightInTime,
                            bool oneTightOutOfTime)
    : L1Candidate(math::PtEtaPhiMLorentzVector{0., 0., 0., 0.}, 0., 0., 0., 0, 0),
      // in this object it makes more sense to the different shower types to
      // the 4 bits, so that the object easily interfaces with the uGT emulator
      oneNominalInTime_(oneNominalInTime),
      oneTightInTime_(oneTightInTime),
      musOutOfTime0_(false),
      musOutOfTime1_(false) {}

l1t::MuonShower::~MuonShower() {}

bool l1t::MuonShower::isValid() const {
  return oneNominalInTime_ or oneTightInTime_ or musOutOfTime0_ or musOutOfTime1_;
}

bool l1t::MuonShower::operator==(const l1t::MuonShower& rhs) const {
  return (oneNominalInTime_ == rhs.isOneNominalInTime() and oneTightInTime_ == rhs.isOneTightInTime() and
          musOutOfTime0_ == rhs.musOutOfTime0() and musOutOfTime1_ == rhs.musOutOfTime1());
}

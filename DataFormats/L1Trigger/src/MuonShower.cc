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
      mus0_(oneNominalInTime),
      mus1_(oneTightInTime),
      musOutOfTime0_(false),
      musOutOfTime1_(false) {}

l1t::MuonShower::~MuonShower() {}

bool l1t::MuonShower::isValid() const { return mus0_ or mus1_ or musOutOfTime0_ or musOutOfTime1_; }

bool l1t::MuonShower::operator==(const l1t::MuonShower& rhs) const {
  return (mus0_ == rhs.mus0() and mus1_ == rhs.mus1() and musOutOfTime0_ == rhs.musOutOfTime0() and
          musOutOfTime1_ == rhs.musOutOfTime1());
}

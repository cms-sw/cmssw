#include "DataFormats/L1Trigger/interface/MuonShower.h"

namespace l1t::io_v1 {
  MuonShower::MuonShower(bool oneNominalInTime,
                         bool oneNominalOutOfTime,
                         bool twoLooseInTime,
                         bool twoLooseOutOfTime,
                         bool oneTightInTime,
                         bool oneTightOutOfTime,
                         bool twoLooseDiffSectorsInTime)
      : L1Candidate(math::PtEtaPhiMLorentzVector{0., 0., 0., 0.}, 0., 0., 0., 0, 0),
        // in this object it makes more sense to the different shower types to
        // the 4 bits, so that the object easily interfaces with the uGT emulator
        oneNominalInTime_(oneNominalInTime),
        oneTightInTime_(oneTightInTime),
        twoLooseDiffSectorsInTime_(twoLooseDiffSectorsInTime),
        musOutOfTime0_(false),
        musOutOfTime1_(false) {}

  MuonShower::~MuonShower() {}

  bool MuonShower::isValid() const {
    return oneNominalInTime_ or oneTightInTime_ or twoLooseDiffSectorsInTime_ or musOutOfTime0_ or musOutOfTime1_;
  }

  bool MuonShower::operator==(const MuonShower& rhs) const {
    return (oneNominalInTime_ == rhs.isOneNominalInTime() and oneTightInTime_ == rhs.isOneTightInTime() and
            musOutOfTime0_ == rhs.musOutOfTime0() and musOutOfTime1_ == rhs.musOutOfTime1() and
            twoLooseDiffSectorsInTime_ == rhs.isTwoLooseDiffSectorsInTime());
  }
}  // namespace l1t::io_v1

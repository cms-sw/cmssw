#include "DataFormats/FTLRecHit/interface/FTLUncalibratedRecHit.h"
#include <cmath>
#include <limits>

FTLUncalibratedRecHit::FTLUncalibratedRecHit()
    : FTLUncalibratedRecHit(
          DetId(), 0, 0, {-1.f, -1.f}, {-1.f, -1.f}, -1.f, -1.f, -1.f, std::numeric_limits<unsigned char>::max()) {}

FTLUncalibratedRecHit::FTLUncalibratedRecHit(const DetId& id,
                                             uint8_t row,
                                             uint8_t column,
                                             std::pair<float, float> ampl,
                                             std::pair<float, float> time,
                                             float timeError,
                                             float position,
                                             float positionError,
                                             unsigned char flags)
    : amplitude_(ampl),
      time_(time),
      timeError_(timeError),
      position_(position),
      positionError_(positionError),
      id_(id),
      row_(row),
      column_(column),
      flags_(flags) {}

FTLUncalibratedRecHit::FTLUncalibratedRecHit(const DetId& id,
                                             std::pair<float, float> ampl,
                                             std::pair<float, float> time,
                                             float timeError,
                                             float position,
                                             float positionError,
                                             unsigned char flags)
    : FTLUncalibratedRecHit(id, 0, 0, ampl, time, timeError, position, positionError, flags) {}

bool FTLUncalibratedRecHit::isSaturated() const { return FTLUncalibratedRecHit::checkFlag(kSaturated); }

bool FTLUncalibratedRecHit::isTimeValid() const {
  if (timeError() < 0)
    return false;
  else
    return true;
}

bool FTLUncalibratedRecHit::isTimeErrorValid() const {
  if (!isTimeValid())
    return false;
  if (timeError() >= 10000)
    return false;

  return true;
}

void FTLUncalibratedRecHit::setFlagBit(FTLUncalibratedRecHit::Flags flag) {
  if (flag == kGood) {
    //then set all bits to zero;
    flags_ = 0;
    return;
  }
  // else set the flagbit
  flags_ |= 0x1 << flag;
}

bool FTLUncalibratedRecHit::checkFlag(FTLUncalibratedRecHit::Flags flag) const {
  if (flag == kGood) {
    if (!flags_)
      return true;
    else
      return false;
  }  // if all flags are unset, then hit is good
  return flags_ & (0x1 << flag);
}

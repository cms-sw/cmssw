#include "TkMeasurementDetSet.h"

void StMeasurementConditionSet::init(int size) {
  activeThisPeriod_.resize(size, true);
  id_.resize(size);
  subId_.resize(size);
  totalStrips_.resize(size);

  bad128Strip_.resize(size * 6);
  hasAny128StripBad_.resize(size);
  badStripBlocks_.resize(size);
}

void StMeasurementConditionSet::set128StripStatus(int i, bool good, int idx) {
  int offset = nbad128 * i;
  if (idx == -1) {
    std::fill(bad128Strip_.begin() + offset, bad128Strip_.begin() + offset + 6, !good);
    hasAny128StripBad_[i] = !good;
  } else {
    bad128Strip_[offset + idx] = !good;
    if (good == false) {
      hasAny128StripBad_[i] = false;
    } else {  // this should not happen, as usually you turn on all fibers
              // and then turn off the bad ones, and not vice-versa,
              // so I don't care if it's not optimized
      hasAny128StripBad_[i] = true;
      for (int j = 0; i < (totalStrips_[j] >> 7); j++) {
        if (bad128Strip_[j + offset] == false) {
          hasAny128StripBad_[i] = false;
          break;
        }
      }
    }
  }
}

void PxMeasurementConditionSet::init(int size) {
  activeThisPeriod_.resize(size, true);
  id_.resize(size);
}

void Phase2OTMeasurementConditionSet::init(int size) {
  activeThisPeriod_.resize(size, true);
  id_.resize(size);
}

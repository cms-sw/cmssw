#include "CondFormats/ESObjects/interface/ESThresholds.h"

ESThresholds::ESThresholds() {
  ts2_ = 0.;
  zs_ = 0.;
}

ESThresholds::ESThresholds(const float& ts2, const float& zs) {
  ts2_ = ts2;
  zs_ = zs;
}

ESThresholds::~ESThresholds() {}

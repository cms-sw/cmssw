#include "CondFormats/ESObjects/interface/ESTimeSampleWeights.h"

ESTimeSampleWeights::ESTimeSampleWeights() {
  w0_ = 0.;
  w1_ = 0.;
  w2_ = 0.;
}

ESTimeSampleWeights::ESTimeSampleWeights(const float& w0, const float& w1, const float& w2) {
  w0_ = w0;
  w1_ = w1;
  w2_ = w2;
}

ESTimeSampleWeights::~ESTimeSampleWeights() {}

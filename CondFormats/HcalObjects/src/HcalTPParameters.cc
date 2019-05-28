#include <algorithm>
#include <iostream>
#include <set>

#include "CondFormats/HcalObjects/interface/HcalTPParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalTPParameters::HcalTPParameters() : version_(0), adcCut_(0), tdcMask_(0), tbits_(0), auxi1_(0), auxi2_(0) {}

HcalTPParameters::~HcalTPParameters() {}

void HcalTPParameters::loadObject(int version, int adcCut, uint64_t tdcMask, uint32_t tbits, int auxi1, int auxi2) {
  version_ = version;
  adcCut_ = adcCut;
  tdcMask_ = tdcMask;
  tbits_ = tbits;
  auxi1_ = auxi1;
  auxi2_ = auxi2;
}

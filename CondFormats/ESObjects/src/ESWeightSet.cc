#include "CondFormats/ESObjects/interface/ESWeightSet.h"
ESWeightSet::ESWeightSet() {}

ESWeightSet::ESWeightSet(const ESWeightSet& rhs) { wgtBeforeSwitch_ = rhs.wgtBeforeSwitch_; }

ESWeightSet::ESWeightSet(ESWeightMatrix& rhs) { wgtBeforeSwitch_ = rhs; }

ESWeightSet& ESWeightSet::operator=(const ESWeightSet& rhs) {
  wgtBeforeSwitch_ = rhs.wgtBeforeSwitch_;
  return *this;
}

ESWeightSet::~ESWeightSet() {}

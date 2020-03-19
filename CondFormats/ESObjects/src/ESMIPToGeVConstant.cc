#include "CondFormats/ESObjects/interface/ESMIPToGeVConstant.h"

ESMIPToGeVConstant::ESMIPToGeVConstant() {
  ESvaluelow_ = 0.;
  ESvaluehigh_ = 0.;
}

ESMIPToGeVConstant::ESMIPToGeVConstant(const float& ESvaluelow, const float& ESvaluehigh) {
  ESvaluelow_ = ESvaluelow;
  ESvaluehigh_ = ESvaluehigh;
}

ESMIPToGeVConstant::~ESMIPToGeVConstant() {}

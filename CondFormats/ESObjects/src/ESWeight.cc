#include "CondFormats/ESObjects/interface/ESWeight.h"

ESWeight::ESWeight() {
  wgt_ = 0.0;
}

ESWeight::ESWeight(const double& awgt) {
  wgt_ = awgt;
}

ESWeight::ESWeight(const ESWeight& awgt) {
  wgt_ = awgt.wgt_;
}

ESWeight& ESWeight::operator=(const ESWeight&rhs) {
   wgt_ = rhs.wgt_;
   return *this;
}

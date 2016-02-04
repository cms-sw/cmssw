/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: EcalWeight.cc,v 1.2 2006/02/23 16:56:35 rahatlou Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalWeight.h"

EcalWeight::EcalWeight() {
  wgt_ = 0.0;
}

EcalWeight::EcalWeight(const double& awgt) {
  wgt_ = awgt;
}

EcalWeight::EcalWeight(const EcalWeight& awgt) {
  wgt_ = awgt.wgt_;
}

EcalWeight& EcalWeight::operator=(const EcalWeight&rhs) {
   wgt_ = rhs.wgt_;
   return *this;
}

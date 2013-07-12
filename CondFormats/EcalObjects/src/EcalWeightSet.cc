/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: EcalWeightSet.cc,v 1.3 2006/02/23 16:56:35 rahatlou Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
EcalWeightSet::EcalWeightSet() {

}

EcalWeightSet::EcalWeightSet(const EcalWeightSet& rhs) {
  wgtBeforeSwitch_ = rhs.wgtBeforeSwitch_;
  wgtAfterSwitch_ = rhs.wgtAfterSwitch_;
  wgtChi2BeforeSwitch_ = rhs.wgtChi2BeforeSwitch_;
  wgtChi2AfterSwitch_ = rhs.wgtChi2AfterSwitch_;

}

EcalWeightSet& EcalWeightSet::operator=(const EcalWeightSet& rhs) {

  wgtBeforeSwitch_ = rhs.wgtBeforeSwitch_;
  wgtAfterSwitch_ = rhs.wgtAfterSwitch_;
  wgtChi2BeforeSwitch_ = rhs.wgtChi2BeforeSwitch_;
  wgtChi2AfterSwitch_ = rhs.wgtChi2AfterSwitch_;
  return *this;
}

EcalWeightSet::~EcalWeightSet() {
}

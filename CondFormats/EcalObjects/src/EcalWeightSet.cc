#include "CondFormats/EcalObjects/interface/EcalWeight.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
//
// defualt ctor creates vectors of length EBDataFrame::MAXSAMPLES==10
//
EcalWeightSet::EcalWeightSet() {

}
EcalWeightSet::EcalWeightSet(const EcalWeightSet& aset) {
}

EcalWeightSet& EcalWeightSet::operator=(const EcalWeightSet& rhs) {
  wgtBeforeSwitch_.clear();
  for(size_t i=0; i<rhs.wgtBeforeSwitch_.size(); ++i) {
    wgtBeforeSwitch_.push_back(rhs.wgtBeforeSwitch_[i]);
  }

  wgtAfterSwitch_.clear();
  for(size_t i=0; i<rhs.wgtAfterSwitch_.size(); ++i) {
    wgtAfterSwitch_.push_back(rhs.wgtAfterSwitch_[i]);
  }

  wgtChi2BeforeSwitch_.clear();
  for(size_t i=0; i<rhs.wgtChi2BeforeSwitch_.size(); ++i) {
    wgtChi2BeforeSwitch_.push_back(rhs.wgtChi2BeforeSwitch_[i]);
  }

  wgtChi2AfterSwitch_.clear();
  for(size_t i=0; i<rhs.wgtChi2AfterSwitch_.size(); ++i) {
    wgtChi2AfterSwitch_.push_back(rhs.wgtChi2AfterSwitch_[i]);
  }

  return *this;
}

EcalWeightSet::~EcalWeightSet() {
}

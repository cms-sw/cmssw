#include "CondFormats/HcalObjects/interface/HcalDcsValue.h"
#include "DataFormats/HcalDetId/interface/HcalDcsDetId.h"

HcalDcsValue::HcalDcsValue () : 
  mId(0), mLS(0), mValue(0), mUpperLimit(0), mLowerLimit(0) {
}

HcalDcsValue::HcalDcsValue (uint32_t fid, int ls, float val, 
			    float upper, float lower) :
  mId(fid), mLS(ls), mValue(val), mUpperLimit(upper), mLowerLimit(lower) {
}

HcalDcsValue::HcalDcsValue (HcalDcsValue const& other) :
  mId(other.mId), mLS(other.mLS), mValue(other.mValue), 
  mUpperLimit(other.mUpperLimit), mLowerLimit(other.mLowerLimit) {
}

HcalDcsValue::~HcalDcsValue () {
}

HcalOtherSubdetector HcalDcsValue::getSubdetector () const {
  HcalDcsDetId tmpId(mId);
  return tmpId.subdet();
}

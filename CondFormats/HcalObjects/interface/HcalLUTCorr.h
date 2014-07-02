#ifndef HcalLUTCorr_h
#define HcalLUTCorr_h

#include "CondFormats/Serialization/interface/Serializable.h"

/*
\class HcalLUTCorr
\author Radek Ofierzynski
contains one LUT correction factor value + corresponding DetId
*/

class HcalLUTCorr
{
 public:
  HcalLUTCorr(): mId(0), mValue(0) {}
  HcalLUTCorr(unsigned long fid, float value): mId(fid), mValue(value) {}

  uint32_t rawId() const {return mId;}

  float getValue() const {return mValue;}

 private:
  uint32_t mId;
  float mValue;

 COND_SERIALIZABLE;
};

#endif

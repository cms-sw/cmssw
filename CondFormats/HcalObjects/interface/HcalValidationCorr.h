#ifndef HcalValidationCorr_h
#define HcalValidationCorr_h

#include "CondFormats/Serialization/interface/Serializable.h"

/*
\class HcalValidationCorr
\author Gena Kukartsev kukarzev@fnal.gov
contains one validation correction factor value + corresponding DetId
*/

class HcalValidationCorr
{
 public:
  HcalValidationCorr(): mId(0), mValue(0) {}
  HcalValidationCorr(unsigned long fid, float value): mId(fid), mValue(value) {}

  uint32_t rawId() const {return mId;}

  float getValue() const {return mValue;}

 private:
  uint32_t mId;
  float mValue;

 COND_SERIALIZABLE;
};

#endif

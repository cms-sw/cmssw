#ifndef HcalZDCLowGainFraction_h
#define HcalZDCLowGainFraction_h

/** 
\class HcalZDCLowGainFraction
\author Audrius Mecionis
POOL object to store lowGainFrac values
*/

#include "CondFormats/Serialization/interface/Serializable.h"
#include <cstdint>

class HcalZDCLowGainFraction {
public:
  HcalZDCLowGainFraction() : mId(0), mValue(0) {}

  HcalZDCLowGainFraction(unsigned long fid, float value) : mId(fid), mValue(value) {}

  uint32_t rawId() const { return mId; }

  float getValue() const { return mValue; }

private:
  uint32_t mId;
  float mValue;

  COND_SERIALIZABLE;
};

#endif

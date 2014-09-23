#ifndef ZDCLowGainFraction_h
#define ZDCLowGainFraction_h

/** 
\class ZDCLowGainFraction
\author Audrius Mecionis
POOL object to store lowGainFrac values
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include <boost/cstdint.hpp>
#include <vector>

class ZDCLowGainFraction {
 public:
  ZDCLowGainFraction() : mId(0), mValue(0) {}

  ZDCLowGainFraction(unsigned long fid, float value): mId(fid), mValue(value) {}

  uint32_t rawId() const {return mId;}

  float getValue() const {return mValue;}

 private:
  uint32_t mId;
  float mValue;

 COND_SERIALIZABLE;
};


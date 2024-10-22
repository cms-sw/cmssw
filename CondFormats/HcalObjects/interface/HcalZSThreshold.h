#ifndef HcalZSThreshold_h
#define HcalZSThreshold_h

/*
\class HcalZSThreshold
\author Radek Ofierzynski
contains one threshold + corresponding DetId
*/

#include "CondFormats/Serialization/interface/Serializable.h"
#include <cstdint>

class HcalZSThreshold {
public:
  HcalZSThreshold() : mId(0), mLevel(0) {}
  HcalZSThreshold(unsigned long fid, int level) : mId(fid), mLevel(level) {}

  uint32_t rawId() const { return mId; }

  int getValue() const { return mLevel; }

private:
  uint32_t mId;
  int mLevel;

  COND_SERIALIZABLE;
};

#endif

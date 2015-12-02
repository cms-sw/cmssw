#ifndef HcalQIEType_h
#define HcalQIEType_h

#include "CondFormats/Serialization/interface/Serializable.h"

/*
\class HcalQIEType
\author Walter Alda 
contains the QIE Typese + corresponding DetId
*/

class HcalQIEType
{
 public:
  HcalQIEType(): mId(0), mValue(0) {}
  HcalQIEType(unsigned long fid, int value): mId(fid), mValue(value) {}

  uint32_t rawId() const {return mId;}

  int getValue() const {return mValue;}

 private:
  uint32_t mId;
  int mValue;

 COND_SERIALIZABLE;
};

#endif

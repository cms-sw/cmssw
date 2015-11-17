#ifndef HcalQIETypes_h
#define HcalQIETypes_h

#include "CondFormats/Serialization/interface/Serializable.h"

/*
\class HcalQIETypes
\author Walter Alda 
contains the QIE Typese + corresponding DetId
*/

class HcalQIETypes
{
 public:
  HcalQIETypes(): mId(0), mValue(0) {}
  HcalQIETypes(unsigned long fid, int value): mId(fid), mValue(value) {}

  uint32_t rawId() const {return mId;}

  int getValue() const {return mValue;}

 private:
  uint32_t mId;
  int mValue;

 COND_SERIALIZABLE;
};

#endif

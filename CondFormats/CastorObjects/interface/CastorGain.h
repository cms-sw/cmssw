#ifndef CastorGain_h
#define CastorGain_h

/** 
\class CastoGain
\author Panos Katsas (UoA)
POOL object to store Gain values 4xCapId
*/

#include "CondFormats/Serialization/interface/Serializable.h"
#include <cstdint>

class CastorGain {
public:
  /// get value for all capId = 0..3
  const float* getValues() const { return &mValue0; }
  /// get value for capId = 0..3
  float getValue(int fCapId) const { return *(getValues() + fCapId); }

  // functions below are not supposed to be used by consumer applications

  CastorGain() : mId(0), mValue0(0), mValue1(0), mValue2(0), mValue3(0) {}

  CastorGain(unsigned long fId, float fCap0, float fCap1, float fCap2, float fCap3)
      : mId(fId), mValue0(fCap0), mValue1(fCap1), mValue2(fCap2), mValue3(fCap3) {}

  uint32_t rawId() const { return mId; }

private:
  uint32_t mId;
  float mValue0;
  float mValue1;
  float mValue2;
  float mValue3;

  COND_SERIALIZABLE;
};

#endif

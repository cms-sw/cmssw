#ifndef HcalL1TriggerObject_h
#define HcalL1TriggerObject_h

/*
\class HcalL1TriggerObject
\author Radek Ofierzynski

*/

#include "CondFormats/Serialization/interface/Serializable.h"
#include <cstdint>

class HcalL1TriggerObject {
public:
  HcalL1TriggerObject() : mId(0), mAvrgPed(0.), mRespCorrGain(0.), mFlag(0) {}

  HcalL1TriggerObject(unsigned long fId, float fAvrgPed, float fRespCorrGain, unsigned long fFlag = 0)
      : mId(fId), mAvrgPed(fAvrgPed), mRespCorrGain(fRespCorrGain), mFlag(fFlag) {}

  uint32_t rawId() const { return mId; }

  float getPedestal() const { return mAvrgPed; }
  float getRespGain() const { return mRespCorrGain; }
  uint32_t getFlag() const { return mFlag; }

private:
  uint32_t mId;
  float mAvrgPed;
  float mRespCorrGain;
  uint32_t mFlag;

  COND_SERIALIZABLE;
};

#endif

#ifndef CastorRawGain_h
#define CastorRawGain_h

/** 
\class CastorRawGain
\author Panos Katsas (UoA)
POOL object to store raw Gain values
*/
#include <string>
#include <cstdint>

class CastorRawGain {
public:
  enum Status { GOOD = 0, BAD = 1 };
  float getValue() const { return mValue; }
  float getError() const { return mError; }
  float getVoltage() const { return mVoltage; }
  Status getStatus() const { return Status(mStatus); }
  std::string strStatus() const { return getStatus() == GOOD ? "GOOD" : "BAD"; }

  CastorRawGain(unsigned long fId = 0) : mId(fId), mValue(0), mError(0), mVoltage(0), mStatus(int(BAD)) {}

  CastorRawGain(unsigned long fId, float fValue, float fError, float fVoltage, Status fStatus)
      : mId(fId), mValue(fValue), mError(fError), mVoltage(fVoltage), mStatus(int(fStatus)) {}

  uint32_t rawId() const { return mId; }

private:
  uint32_t mId;
  float mValue;
  float mError;
  float mVoltage;
  int mStatus;
};

#endif

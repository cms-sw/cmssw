#ifndef CastorSaturationCorr_h
#define CastorSaturationCorr_h

/** 
\class CastorSaturationCorr
\author adapted for CASTOR by Hans Van Haevermaet
POOL object to store saturation correction values
*/

#include "CondFormats/Serialization/interface/Serializable.h"
#include <cstdint>

class CastorSaturationCorr {
public:
  CastorSaturationCorr() : mId(0), mSatCorr(0) {}

  CastorSaturationCorr(unsigned long fId, float fSatCorr) : mId(fId), mSatCorr(fSatCorr) {}

  uint32_t rawId() const { return mId; }

  float getValue() const { return mSatCorr; }

private:
  uint32_t mId;
  float mSatCorr;

  COND_SERIALIZABLE;
};

#endif

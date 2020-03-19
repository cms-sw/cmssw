#ifndef CastorRecoParam_h
#define CastorRecoParam_h

/** 
\class CastorRecoParam
\author Radek Ofierzynski - adapted for CASTOR by Hans Van Haevermaet
POOL object to store timeslice reco values
*/

#include "CondFormats/Serialization/interface/Serializable.h"
#include <cstdint>

class CastorRecoParam {
public:
  CastorRecoParam() : mId(0), mFirstSample(0), mSamplesToAdd(0) {}

  CastorRecoParam(unsigned long fId, unsigned int fFirstSample, unsigned int fSamplesToAdd)
      : mId(fId), mFirstSample(fFirstSample), mSamplesToAdd(fSamplesToAdd) {}

  uint32_t rawId() const { return mId; }

  unsigned int firstSample() const { return mFirstSample; }
  unsigned int samplesToAdd() const { return mSamplesToAdd; }

private:
  uint32_t mId;
  uint32_t mFirstSample;
  uint32_t mSamplesToAdd;

  COND_SERIALIZABLE;
};

#endif

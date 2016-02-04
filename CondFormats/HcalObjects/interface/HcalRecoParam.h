#ifndef HcalRecoParam_h
#define HcalRecoParam_h

/** 
\class HcalRecoParam
\author Radek Ofierzynski
POOL object to store timeslice reco values
*/

#include <boost/cstdint.hpp>

class HcalRecoParam {
 public:
  HcalRecoParam():mId(0), mFirstSample(0), mSamplesToAdd(0) {}

  HcalRecoParam(unsigned long fId, unsigned int fFirstSample, unsigned int fSamplesToAdd):
    mId(fId), mFirstSample(fFirstSample), mSamplesToAdd(fSamplesToAdd) {}

  uint32_t rawId () const {return mId;}

  unsigned int firstSample() const {return mFirstSample;}
  unsigned int samplesToAdd() const {return mSamplesToAdd;}

 private:
  uint32_t mId;
  uint32_t mFirstSample;
  uint32_t mSamplesToAdd;
};

#endif

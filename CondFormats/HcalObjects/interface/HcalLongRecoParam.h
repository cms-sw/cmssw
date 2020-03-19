#ifndef HcalLongRecoParam_h
#define HcalLongRecoParam_h

/** 
\class HcalLongRecoParam
\author Radek Ofierzynski
POOL object to store timeslice reco values, long version (for ZDC)
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <cstdint>

class HcalLongRecoParam {
public:
  HcalLongRecoParam() : mId(0) {}

  HcalLongRecoParam(unsigned long fId,
                    const std::vector<unsigned int>& fSignalTS,
                    const std::vector<unsigned int>& fNoiseTS)
      : mId(fId), mSignalTS(fSignalTS), mNoiseTS(fNoiseTS) {}

  uint32_t rawId() const { return mId; }

  std::vector<unsigned int> signalTS() const { return mSignalTS; }
  std::vector<unsigned int> noiseTS() const { return mNoiseTS; }

private:
  uint32_t mId;
  std::vector<uint32_t> mSignalTS;
  std::vector<uint32_t> mNoiseTS;

  COND_SERIALIZABLE;
};

#endif

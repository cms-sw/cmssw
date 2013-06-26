#ifndef HcalLongRecoParam_h
#define HcalLongRecoParam_h

/** 
\class HcalLongRecoParam
\author Radek Ofierzynski
POOL object to store timeslice reco values, long version (for ZDC)
*/

#include <boost/cstdint.hpp>
#include <vector>

class HcalLongRecoParam {
 public:
  HcalLongRecoParam():mId(0) {}

  HcalLongRecoParam(unsigned long fId, std::vector<unsigned int> fSignalTS, std::vector<unsigned int> fNoiseTS):
    mId(fId), mSignalTS(fSignalTS), mNoiseTS(fNoiseTS) {}

  uint32_t rawId () const {return mId;}

  std::vector<unsigned int> signalTS() const {return mSignalTS;}
  std::vector<unsigned int> noiseTS() const {return mNoiseTS;}

 private:
  uint32_t mId;
  std::vector<uint32_t> mSignalTS;
  std::vector<uint32_t> mNoiseTS;
};

#endif

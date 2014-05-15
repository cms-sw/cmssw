#ifndef HcalRecoParam_h
#define HcalRecoParam_h

/** 
\class HcalRecoParam
\author Radek Ofierzynski
POOL object to store timeslice reco values

mParam1, mParam2 re-define to keep more parameters   28-Oct-2011  sk.
 
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include <boost/cstdint.hpp>

class HcalRecoParam {
 public:
  HcalRecoParam():mId(0), mParam1(0), mParam2(0) {}

  HcalRecoParam(unsigned long fId, unsigned int fParam1, unsigned int fParam2):
    mId(fId), mParam1(fParam1), mParam2(fParam2) {}

  uint32_t rawId () const {return mId;}

  unsigned int param1() const {return mParam1;}
  unsigned int param2() const {return mParam2;}

  bool correctForPhaseContainment() const {return mParam1&0x1;}
  bool correctForLeadingEdge() const {return (mParam1>>1)&0x1;}
  float correctionPhaseNS() const {return ((mParam1>>2)&0xFF)/4.-32.;}
  unsigned int firstSample() const {return  (mParam1<10)?(mParam1):((mParam1>>10)&0xF);}
  unsigned int samplesToAdd() const {return (mParam1<10)?(mParam2):((mParam1>>14)&0xF);}
  unsigned int pulseShapeID() const {return (mParam1>>18)&0x1FF;}

  bool useLeakCorrection () const {return mParam2&0x1;}
  unsigned int leakCorrectionID () const {return (mParam2>>1)&0xF;}
  bool correctForTimeslew () const {return (mParam2>>5)&0x1;}
  unsigned int timeslewCorrectionID () const {return (mParam2>>6)&0xF;}
  bool correctTiming () const {return (mParam2>>10)&0x1;}
  unsigned int firstAuxTS () const { return (mParam2>>11)&0xF;}
  unsigned int specialCaseID () const {return (mParam2>>15)&0xF;}
  unsigned int noiseFlaggingID () const {return (mParam2>>19)&0xF;}
  unsigned int pileupCleaningID () const {return (mParam2>>23)&0xF;}
  unsigned int packingScheme () const {return (mParam2>>27)&0xF;}

 private:
  uint32_t mId;
  uint32_t mParam1;
  uint32_t mParam2;

 COND_SERIALIZABLE;
};

#endif

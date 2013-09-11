// -*- C++ -*-
#ifndef HcalDcsValue_h
#define HcalDcsValue_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <stdint.h>
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

class HcalDcsValue {
public:

  HcalDcsValue ();
  HcalDcsValue (uint32_t fid, int ls, float val, float upper,
		float lower);
  HcalDcsValue (HcalDcsValue const& other);

  virtual ~HcalDcsValue ();
  
  uint32_t DcsId () const { return mId; }
  int LS () const { return mLS; }
  float getValue () const { return mValue;}
  float getUpperLimit () const { return mUpperLimit; }
  float getLowerLimit () const { return mLowerLimit; }
  bool isValueGood () const { 
    return ((mValue <= mUpperLimit) && (mValue >= mLowerLimit)); 
  }

  HcalOtherSubdetector getSubdetector() const;

  bool operator < (HcalDcsValue const& rhs) const {
    if (mId == rhs.mId) return (mLS < rhs.mLS);
    return (mId < rhs.mId);
  }

private:
  uint32_t mId;
  int mLS;
  float mValue;
  float mUpperLimit;
  float mLowerLimit;

  COND_SERIALIZABLE;
};

#endif

#ifndef CastorGainWidth_h
#define CastorGainWidth_h

/** 
\class CastorGainWidth
\author Panos Katsas (UoA)
POOL object to store GainWidth values 4xCapId
*/
#include <boost/cstdint.hpp>

class CastorGainWidth {
 public:
  /// get value for all capId = 0..3
  const float* getValues () const {return &mValue0;}
  /// get value for capId = 0..3
  float getValue (int fCapId) const {return *(getValues () + fCapId);}

  // functions below are not supposed to be used by consumer applications

  CastorGainWidth () : mId (0), mValue0 (0) {}

  CastorGainWidth (unsigned long fId, float fCap0, float fCap1, float fCap2, float fCap3) :
    mId (fId),
    mValue0 (fCap0) {}

  uint32_t rawId () const {return mId;}

 private:
  uint32_t mId;
  float mValue0;
};

#endif

#ifndef CastorPedestal_h
#define CastorPedestal_h

/** 
\class CastorPedestal
\author Fedor Ratnikov (UMd)
POOL object to store Pedestal values 4xCapId
$Author: ratnikov
$Date: 2009/03/24 16:05:27 $
$Revision: 1.8 $
Adapted for CASTOR by L. Mundim (26/03/2009)
*/
#include "CondFormats/Serialization/interface/Serializable.h"
#include <cstdint>

class CastorPedestal {
public:
  /// get value for all capId = 0..3
  const float* getValues() const { return &mValue0; }
  /// get value for capId = 0..3
  float getValue(int fCapId) const { return *(getValues() + fCapId); }

  /// get width for all capId = 0..3
  const float* getWidths() const { return &mWidth0; }
  /// get width for capId = 0..3
  float getWidth(int fCapId) const { return *(getWidths() + fCapId); }

  // functions below are not supposed to be used by consumer applications

  CastorPedestal()
      : mId(0), mValue0(0), mValue1(0), mValue2(0), mValue3(0), mWidth0(0), mWidth1(0), mWidth2(0), mWidth3(0) {}

  CastorPedestal(unsigned long fId,
                 float fCap0,
                 float fCap1,
                 float fCap2,
                 float fCap3,
                 float wCap0 = 0,
                 float wCap1 = 0,
                 float wCap2 = 0,
                 float wCap3 = 0)
      : mId(fId),
        mValue0(fCap0),
        mValue1(fCap1),
        mValue2(fCap2),
        mValue3(fCap3),
        mWidth0(wCap0),
        mWidth1(wCap1),
        mWidth2(wCap2),
        mWidth3(wCap3) {}

  uint32_t rawId() const { return mId; }

private:
  uint32_t mId;
  float mValue0;
  float mValue1;
  float mValue2;
  float mValue3;
  float mWidth0;
  float mWidth1;
  float mWidth2;
  float mWidth3;

  COND_SERIALIZABLE;
};

#endif

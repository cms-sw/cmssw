#ifndef CastorQIECoder_h
#define CastorQIECoder_h

/** 
\class CastorQIECoder
\author Fedor Ratnikov (UMd)
POOL object to store QIE coder parameters for one channel
$Author: ratnikov
$Date: 2008/03/05 10:38:03 $
$Revision: 1.9 $
Modified for CASTOR by L. Mundim
*/
#include <boost/cstdint.hpp>

#include <vector>
#include <algorithm>

class CastorQIEShape;

class CastorQIECoder {
 public:
  CastorQIECoder (unsigned long fId = 0) : mId (fId), mOffset00(0), mSlope00(0) {};

  /// ADC [0..127] + capid [0..3] -> fC conversion
  float charge (const CastorQIEShape& fShape, unsigned fAdc, unsigned fCapId) const;
  /// fC + capid [0..3] -> ADC conversion
  unsigned adc (const CastorQIEShape& fShape, float fCharge, unsigned fCapId) const;

  // following methods are not for use by consumers
  float offset (unsigned fCapId, unsigned fRange) const;
  float slope (unsigned fCapId, unsigned fRange) const;

  void setOffset (unsigned fCapId, unsigned fRange, float fValue);
  void setSlope (unsigned fCapId, unsigned fRange, float fValue);

  uint32_t rawId () const {return mId;}

 private:
  uint32_t mId;
  float mOffset00;
  float mSlope00;
};

#endif

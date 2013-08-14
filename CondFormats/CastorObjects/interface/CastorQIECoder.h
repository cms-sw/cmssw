#ifndef CastorQIECoder_h
#define CastorQIECoder_h

/** 
\class CastorQIECoder
\author Fedor Ratnikov (UMd)
POOL object to store QIE coder parameters for one channel
$Author: ratnikov
$Date: 2009/03/26 18:03:15 $
$Revision: 1.2 $
Modified for CASTOR by L. Mundim
*/
#include <boost/cstdint.hpp>

#include <vector>
#include <algorithm>

class CastorQIEShape;

class CastorQIECoder {
 public:
  CastorQIECoder (unsigned long fId = 0) : mId (fId), mOffset00(0),   mOffset01(0),  mOffset02(0),
                 mOffset03(0),  mOffset10(0),  mOffset11(0),  mOffset12(0),  mOffset13(0),
                 mOffset20(0),  mOffset21(0),  mOffset22(0),  mOffset23(0),  mOffset30(0),
                 mOffset31(0),  mOffset32(0),  mOffset33(0),  mSlope00(0),  mSlope01(0),  
                 mSlope02(0),  mSlope03(0),  mSlope10(0),  mSlope11(0),  mSlope12(0),  
                 mSlope13(0),  mSlope20(0),  mSlope21(0),  mSlope22(0),  mSlope23(0), 
                 mSlope30(0),  mSlope31(0),  mSlope32(0),  mSlope33(0) {};

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
  float mOffset01; 
  float mOffset02; 
  float mOffset03; 
  float mOffset10; 
  float mOffset11; 
  float mOffset12; 
  float mOffset13; 
  float mOffset20; 
  float mOffset21; 
  float mOffset22; 
  float mOffset23; 
  float mOffset30; 
  float mOffset31; 
  float mOffset32; 
  float mOffset33; 
  float mSlope00; 
  float mSlope01; 
  float mSlope02;
  float mSlope03;
  float mSlope10;
  float mSlope11;
  float mSlope12;
  float mSlope13;
  float mSlope20;
  float mSlope21;
  float mSlope22;
  float mSlope23;
  float mSlope30;
  float mSlope31;
  float mSlope32;
  float mSlope33;
};

#endif

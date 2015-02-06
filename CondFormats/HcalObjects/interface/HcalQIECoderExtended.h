#ifndef HcalQIECoderExtended_h
#define HcalQIECoderExtended_h

/** 
\class HcalQIECoderExtended
\author  Clemencia Mora (CBPF)
Extended version of QIECoder to add QIEId: barcode and channel
$Author: cmora
$Date: 2015/01/13  $
$Revision: 1.0 $
*/

#include "CondFormats/Serialization/interface/Serializable.h"
//#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"

#include <boost/cstdint.hpp>

#include <vector>
#include <algorithm>

class HcalQIEShape;

class HcalQIECoderExtended {
 public:
  HcalQIECoderExtended (unsigned long fId = 0) : mId(fId), mQIEbarcode(0), mQIEchannel(0) { setQIEIndex(0);}

  int getQIEbarcode()const { return mQIEbarcode; }
  int getQIEchannel()const { return mQIEchannel; }  

  /// ADC [0..127] + capid [0..3] -> fC conversion
  float charge (const HcalQIEShape& fShape, unsigned fAdc, unsigned fCapId) const;
  /// fC + capid [0..3] -> ADC conversion
  unsigned adc (const HcalQIEShape& fShape, float fCharge, unsigned fCapId) const;

  // following methods are not for use by consumers
  float offset (unsigned fCapId, unsigned fRange) const;
  float slope (unsigned fCapId, unsigned fRange) const;
  
  void setOffset (unsigned fCapId, unsigned fRange, float fValue);
  void setSlope (unsigned fCapId, unsigned fRange, float fValue);

  uint32_t rawId () const {return mId;}

  uint32_t qieIndex() const {return mQIEIndex;}
  void setQIEIndex(uint32_t v) { mQIEIndex=v;}


  void setQIEId(int bar, int chan);


 private:
  uint32_t mId;
  int mQIEbarcode ;
  int mQIEchannel ;
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
  unsigned int mQIEIndex COND_TRANSIENT;

 COND_SERIALIZABLE;
};

#endif

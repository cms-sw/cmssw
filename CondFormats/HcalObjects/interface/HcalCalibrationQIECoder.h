#ifndef HcalCalibrationQIECoder_h
#define HcalCalibrationQIECoder_h

/** 
\class HcalQIECoder
\author Fedor Ratnikov (UMd)
POOL object to store calibration mode QIE coder parameters for one channel
$Id
*/

#include <vector>
#include <algorithm>
#include <boost/cstdint.hpp>

class HcalCalibrationQIECoder {
 public:
  HcalCalibrationQIECoder (unsigned long fId = 0) : mId (fId) {}
  /// ADC [0..31] -> fC conversion
  float charge (const unsigned fAdc) const;
  /// fC -> ADC conversion
  unsigned adc (const float fCharge) const;

  // following methods are not for use by consumers
  float minCharge (unsigned fBin) const;
  // 32 values 
  const float* minCharges () const;
  void setMinCharge (unsigned fBin, float fValue);
  void setMinCharges (const float fValue [64]);
  uint32_t rawId () const {return mId;}
 private:
  uint32_t mId;
  float bin0;
  float bin1;
  float bin2;
  float bin3;
  float bin4;
  float bin5;
  float bin6;
  float bin7;
  float bin8;
  float bin9;
  float bin10;
  float bin11;
  float bin12;
  float bin13;
  float bin14;
  float bin15;
  float bin16;
  float bin17;
  float bin18;
  float bin19;
  float bin20;
  float bin21;
  float bin22;
  float bin23;
  float bin24;
  float bin25;
  float bin26;
  float bin27;
  float bin28;
  float bin29;
  float bin30;
  float bin31;
  float bin32;
  float bin33;
  float bin34;
  float bin35;
  float bin36;
  float bin37;
  float bin38;
  float bin39;
  float bin40;
  float bin41;
  float bin42;
  float bin43;
  float bin44;
  float bin45;
  float bin46;
  float bin47;
  float bin48;
  float bin49;
  float bin50;
  float bin51;
  float bin52;
  float bin53;
  float bin54;
  float bin55;
  float bin56;
  float bin57;
  float bin58;
  float bin59;
  float bin60;
  float bin61;
  float bin62;
  float bin63;
  const float* base () const {return &bin0;}
  float* base () {return &bin0;}
};

#endif

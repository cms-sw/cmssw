#ifndef CastorCalibrationQIECoder_h
#define CastorCalibrationQIECoder_h

/** 
\class CastorQIECoder
\author Panos Katsas (UoA)
POOL object to store calibration mode QIE coder parameters for one channel
$Id
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <algorithm>
#include <cstdint>

class CastorCalibrationQIECoder {
public:
  CastorCalibrationQIECoder(unsigned long fId = 0) : mId(fId) {}
  /// ADC [0..31] -> fC conversion
  float charge(const unsigned fAdc) const;
  /// fC -> ADC conversion
  unsigned adc(const float fCharge) const;

  // following methods are not for use by consumers
  float minCharge(unsigned fBin) const;
  // 32 values
  const float* minCharges() const;
  void setMinCharge(unsigned fBin, float fValue);
  void setMinCharges(const float fValue[32]);
  uint32_t rawId() const { return mId; }

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
  const float* base() const { return &bin0; }
  float* base() { return &bin0; }

  COND_SERIALIZABLE;
};

#endif

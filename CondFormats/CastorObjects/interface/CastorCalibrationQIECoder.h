#ifndef CastorCalibrationQIECoder_h
#define CastorCalibrationQIECoder_h

/** 
\class CastorQIECoder
\author Panos Katsas (UoA)
POOL object to store calibration mode QIE coder parameters for one channel
$Id
*/

#include <vector>
#include <algorithm>
#include <boost/cstdint.hpp>

class CastorCalibrationQIECoder {
 public:
  CastorCalibrationQIECoder (unsigned long fId = 0) : mId (fId) {}
  /// ADC [0..31] -> fC conversion
  float charge (const unsigned fAdc) const;
  /// fC -> ADC conversion
  unsigned adc (const float fCharge) const;

  // following methods are not for use by consumers
  float minCharge (unsigned fBin) const;
  // 32 values 
  const float* minCharges () const;
  void setMinCharge (unsigned fBin, float fValue);
  void setMinCharges (const float fValue [32]);
  uint32_t rawId () const {return mId;}
 private:
  uint32_t mId;
  float bin0;
  const float* base () const {return &bin0;}
  float* base () {return &bin0;}
};

#endif

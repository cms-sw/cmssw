/** 
\class CastorQIEData
\author Panos Katsas (UoA)
POOL object to store calibration mode QIE coder parameters for one channel
$Id
*/

#include <iostream>

#include "CondFormats/CastorObjects/interface/CastorQIEShape.h"
#include "CondFormats/CastorObjects/interface/CastorCalibrationQIECoder.h"

float CastorCalibrationQIECoder::charge (unsigned fAdc) const {
  const float* data = base ();
  if (fAdc >= 31) return (3*data[31]-data[30])/2.; // extrapolation
  return (data[fAdc]+data[fAdc+1])/2;
}

unsigned CastorCalibrationQIECoder::adc (float fCharge) const {
  const float* data = base ();
  unsigned adc = 1;
  for (; adc < 32; adc++) {
    if (fCharge < data[adc]) return adc-1;
  }
  return 31; // overflow
}

float CastorCalibrationQIECoder::minCharge (unsigned fBin) const {
  const float* data = base ();
  return fBin < 32 ? data[fBin] : data[31];
}

const float* CastorCalibrationQIECoder::minCharges () const {
  const float* data = base ();
  return data;
}


void CastorCalibrationQIECoder::setMinCharge (unsigned fBin, float fValue) {
  float* data = base ();
  if (fBin < 32) data [fBin] = fValue;
}

void CastorCalibrationQIECoder::setMinCharges (const float fValue [32]) {
  float* data = base ();
  for (int i = 0; i < 32; i++) data[i] = fValue[i];
}

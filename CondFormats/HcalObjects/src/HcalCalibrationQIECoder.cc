/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store calibration mode QIE coder parameters for one channel
$Id
*/

#include <iostream>

#include "CondFormats/HcalObjects/interface/HcalCalibrationQIECoder.h"

float HcalCalibrationQIECoder::charge (unsigned fAdc) const {
  const float* data = base ();
  if (fAdc >= 63) return (3*data[63]-data[62])/2.; // extrapolation
  return (data[fAdc]+data[fAdc+1])/2;
}

unsigned HcalCalibrationQIECoder::adc (float fCharge) const {
  const float* data = base ();
  unsigned adc = 1;
  for (; adc < 64; adc++) {
    if (fCharge < data[adc]) return adc-1;
  }
  return 31; // overflow
}

float HcalCalibrationQIECoder::minCharge (unsigned fBin) const {
  const float* data = base ();
  return fBin < 64 ? data[fBin] : data[63];
}

const float* HcalCalibrationQIECoder::minCharges () const {
  const float* data = base ();
  return data;
}


void HcalCalibrationQIECoder::setMinCharge (unsigned fBin, float fValue) {
  float* data = base ();
  if (fBin < 64) data [fBin] = fValue;
}

void HcalCalibrationQIECoder::setMinCharges (const float fValue [64]) {
  float* data = base ();
  for (int i = 0; i < 64; i++) data[i] = fValue[i];
}

/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store pedestal values 4xCapId
$Author: ratnikov
$Date: 2005/12/16 20:56:15 $
$Revision: 1.5 $
*/
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"

namespace {
  const float binMin [32] = {-1,  0,  1,  2,  3,  4,  5,  6,  7,  8,
			     9, 10, 11, 12, 13, 14, 16, 18, 20, 22,
			     24, 26, 28, 31, 34, 37, 40, 44, 48, 52,
			     57, 62};
}

HcalQIEShape::HcalQIEShape() 
{
  for (int i = 0; i < 32; i++) mValues [i] = binMin [i];
  expand ();
}

HcalQIEShape::~HcalQIEShape() {}

void HcalQIEShape::expand () {
  int scale = 1;
  for (unsigned range = 1; range < 4; range++) {
    scale = scale * 5;
    unsigned index = range * 32;
    mValues [index] = mValues [index - 2]; // link to previous range
    for (unsigned i = 1; i < 32; i++) {
      mValues [index + i] =  mValues [index + i - 1] + scale * (mValues [i] - mValues [i - 1]);
    }
  }
  mValues [128] = 2 * mValues [127] - mValues [126]; // extrapolate
}

float HcalQIEShape::lowEdge (unsigned fAdc) const {
  if (fAdc < 128) return mValues [fAdc];
  return 0.;
}

float HcalQIEShape::center (unsigned fAdc) const {
  if (fAdc < 128) {
    if (fAdc % 32 == 31) return 0.5 * (3 * mValues [fAdc] - mValues [fAdc - 1]); // extrapolate
    else       return 0.5 * (mValues [fAdc] + mValues [fAdc + 1]); // interpolate
  }
  return 0.;
}

float HcalQIEShape::highEdge (unsigned fAdc) const {
  if (fAdc < 128) return mValues [fAdc+1];
  return 0.;
}

bool HcalQIEShape::setLowEdge (float fValue, unsigned fAdc) {
  if (fAdc >= 32) return false; 
  mValues [fAdc] = fValue;
  return true;
}

bool HcalQIEShape::setLowEdges (const float fValue [32]) {
  bool result = true;
  for (int adc = 0; adc < 32; adc++) result = result && setLowEdge (fValue [adc], adc);
  expand ();
  return result;
}


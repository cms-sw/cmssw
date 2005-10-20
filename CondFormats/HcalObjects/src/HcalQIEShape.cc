/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store pedestal values 4xCapId
$Author: ratnikov
$Date: 2005/10/18 23:34:56 $
$Revision: 1.1 $
*/

#include <iostream>

#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"

HcalQIEShape::HcalQIEShape() 
: mValues (129, 0.) {}

HcalQIEShape::~HcalQIEShape() {}

float HcalQIEShape::lowEdge (unsigned fAdc) const {
  if (fAdc < 128) return mValues [fAdc];
  return 0.;
}

float HcalQIEShape::highEdge (unsigned fAdc) const {
  if (fAdc < 128) return mValues [fAdc+1];
  return 0.;
}

bool HcalQIEShape::setLowEdge (float fValue, unsigned fAdc) {
  if (fAdc >= 128) return false; 
  mValues [fAdc] = fValue;
  if (fAdc >= 126) mValues [128] = 2 * mValues [127] - mValues [126]; // extrapolate
  return true;
}

bool HcalQIEShape::setLowEdges (const float fValue [128]) {
  bool result = true;
  for (int adc = 0; adc < 128; adc++) result = result || setLowEdge (fValue [adc], adc);
  return result;
}


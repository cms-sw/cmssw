/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store pedestal values 4xCapId
$Author: ratnikov
$Date: 2005/10/20 05:18:37 $
$Revision: 1.2 $
*/

#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"

HcalQIEShape::HcalQIEShape() 
: mValues (33, 0.) {
}

HcalQIEShape::~HcalQIEShape() {}

float HcalQIEShape::lowEdge (unsigned fAdc) const {
  if (fAdc < 32) return mValues [fAdc];
  return 0.;
}

float HcalQIEShape::highEdge (unsigned fAdc) const {
  if (fAdc < 32) return mValues [fAdc+1];
  return 0.;
}

bool HcalQIEShape::setLowEdge (float fValue, unsigned fAdc) {
  if (fAdc >= 32) return false; 
  mValues [fAdc] = fValue;
  if (fAdc >= 30) mValues [32] = 2 * mValues [31] - mValues [30]; // extrapolate
  return true;
}

bool HcalQIEShape::setLowEdges (const float fValue [32]) {
  bool result = true;
  for (int adc = 0; adc < 32; adc++) result = result && setLowEdge (fValue [adc], adc);
  return result;
}


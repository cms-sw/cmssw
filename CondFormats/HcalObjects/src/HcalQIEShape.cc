/* 
\class HcalQIEShape
POOL object to store pedestal values 4xCapId
$Author: andersj
$Date: 2010/06/17 20:56:15 $
$Revision: 1.1 $
*/
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"

namespace {
  const float binMin [64] = {-1,  0,  1,  2,  3,    4,  5,  6,  7,  8,    9, 10, 11, 12, 13,   14,  // 16*1
                             15, 17, 19, 21, 23,   25, 27, 29, 31, 33,   35, 37, 39, 41, 43,   45, 47, 49, 51, 53,//20*2
			     55, 59, 63, 67, 71,   75, 79, 83, 87, 91,   95, 99, 103,107,111, 115,119,123,127,131,  135,//21*4 
                             139, 147, 155, 163, 171, 179, 187};// 7*8
}                                                                                   // total: 64


HcalQIEShape::HcalQIEShape() 
{
  for (int i = 0; i < 64; i++) mValues [i] = binMin [i];
  expand ();
}

HcalQIEShape::~HcalQIEShape() {}

void HcalQIEShape::expand () {
  int scale = 1;
  for (unsigned range = 1; range < 4; range++) {
    scale = scale * 8;
    unsigned index = range * 64;
    mValues [index] = mValues [index - 2]; // link to previous range
    for (unsigned i = 1; i < 64; i++) {
      mValues [index + i] =  mValues [index + i - 1] + scale * (mValues [i] - mValues [i - 1]);
    }
  }
  mValues [256] = 2 * mValues [255] - mValues [254]; // extrapolate
}

float HcalQIEShape::lowEdge (unsigned fAdc) const {
  if (fAdc < 256) return mValues [fAdc];
  return 0.;
}

float HcalQIEShape::center (unsigned fAdc) const {
  if (fAdc < 256) {
    if (fAdc % 64 == 63) return 0.5 * (3 * mValues [fAdc] - mValues [fAdc - 1]); // extrapolate
    else       return 0.5 * (mValues [fAdc] + mValues [fAdc + 1]); // interpolate
  }
  return 0.;
}

float HcalQIEShape::highEdge (unsigned fAdc) const {
  if (fAdc < 256) return mValues [fAdc+1];
  return 0.;
}

bool HcalQIEShape::setLowEdge (float fValue, unsigned fAdc) {
  if (fAdc >= 64) return false; 
  mValues [fAdc] = fValue;
  return true;
}

bool HcalQIEShape::setLowEdges (const float fValue [64]) {
  bool result = true;
  for (int adc = 0; adc < 64; adc++) result = result && setLowEdge (fValue [adc], adc);
  expand ();
  return result;
}

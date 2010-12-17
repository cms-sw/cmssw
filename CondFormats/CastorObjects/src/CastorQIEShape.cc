/** 
\class CastorQIEData
\author Panos Katsas (UoA)
POOL object to store pedestal values 4xCapId
*/
#include "CondFormats/CastorObjects/interface/CastorQIEShape.h"

namespace {
  /*const float binMin [32] = {-1,  0,  1,  2,  3,  4,  5,  6,  7,  8,
			     9, 10, 11, 12, 13, 14, 16, 18, 20, 22,
			     24, 26, 28, 31, 34, 37, 40, 44, 48, 52,
			     57, 62}; */
			     
const float binMin [32] = {-3, -0.4,  2.2,  4.8,  7.4,  10,  12.6, 15.2, 
                           17.8,  20.4, 23, 25.6, 28.2, 30.8 ,33.4, 36,
			   41.2, 46.4, 51.6, 56.8, 62, 67.2, 73, 80.8, 
			   88.6, 96.4, 104, 114.4, 124.8, 135, 148, 161}; 
			     
			   }
			   
CastorQIEShape::CastorQIEShape() 
{
  for (int i = 0; i < 32; i++) mValues [i] = binMin [i];
  expand ();
}

CastorQIEShape::~CastorQIEShape() {}

void CastorQIEShape::expand () {
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

float CastorQIEShape::lowEdge (unsigned fAdc) const {
  if (fAdc < 128) return mValues [fAdc];
  return 0.;
}

float CastorQIEShape::center (unsigned fAdc) const {
  if (fAdc < 128) {
    if (fAdc % 32 == 31) return 0.5 * (3 * mValues [fAdc] - mValues [fAdc - 1]); // extrapolate
    else       return 0.5 * (mValues [fAdc] + mValues [fAdc + 1]); // interpolate
  }
  return 0.;
}

float CastorQIEShape::highEdge (unsigned fAdc) const {
  if (fAdc >= 128 ) return 0;
  if (fAdc == 127 ) return mValues [fAdc+1];
  if (fAdc % 32 == 31) return mValues [fAdc+3];
  return mValues [fAdc+1];
}

bool CastorQIEShape::setLowEdge (float fValue, unsigned fAdc) {
  if (fAdc >= 32) return false; 
  mValues [fAdc] = fValue;
  return true;
}

bool CastorQIEShape::setLowEdges (const float fValue [32]) {
  bool result = true;
  for (int adc = 0; adc < 32; adc++) result = result && setLowEdge (fValue [adc], adc);
  expand ();
  return result;
}

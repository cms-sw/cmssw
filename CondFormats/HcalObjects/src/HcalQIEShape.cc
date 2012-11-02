/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store pedestal values 4xCapId
$Author: ratnikov
$Date: 2005/12/16 20:56:15 $
$Revision: 1.5 $
*/
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"

//namespace {
//  const float binMin [32] = {-1,  0,  1,  2,  3,  4,  5,  6,  7,  8,
//			     9, 10, 11, 12, 13, 14, 16, 18, 20, 22,
//			     24, 26, 28, 31, 34, 37, 40, 44, 48, 52,
			     //			     57, 62};
  //}

HcalQIEShape::HcalQIEShape()
  :nbins_(0)
{
}

HcalQIEShape::~HcalQIEShape() {}

void HcalQIEShape::expand () {
  int scale = 1;
  for (unsigned range = 1; range < 4; range++) {
    scale = scale * 5;
    unsigned index = range * nbins_;
    mValues [index] = mValues [index - 2]; // link to previous range
    for (unsigned i = 1; i < nbins_; i++) {
      mValues [index + i] =  mValues [index + i - 1] + scale * (mValues [i] - mValues [i - 1]);
    }
  }
  mValues [nbins_*4] = 2 * mValues [nbins_*4-1] - mValues [nbins_*4-2]; // extrapolate
}

float HcalQIEShape::lowEdge (unsigned fAdc) const {
  if (fAdc < 4*nbins_) return mValues [fAdc];
  return 0.;
}

float HcalQIEShape::center (unsigned fAdc) const {
  if (fAdc < 4*nbins_) {
    if (fAdc % nbins_ == nbins_-1) return 0.5 * (3 * mValues [fAdc] - mValues [fAdc - 1]); // extrapolate
    else       return 0.5 * (mValues [fAdc] + mValues [fAdc + 1]); // interpolate
  }
  return 0.;
}

float HcalQIEShape::highEdge (unsigned fAdc) const {
  if (fAdc < 4*nbins_) return mValues [fAdc+1];
  return 0.;
}

bool HcalQIEShape::setLowEdge (float fValue, unsigned fAdc) {
  if (fAdc >= nbins_) return false; 
  mValues [fAdc] = fValue;
  return true;
}

bool HcalQIEShape::setLowEdges (unsigned int nbins, const float *fValue) {
  nbins_=nbins;
  mValues.clear();
  mValues.resize(4*nbins_);
  bool result = true;
  for (unsigned int adc = 0; adc < nbins_; adc++) result = result && setLowEdge (fValue [adc], adc);
  expand ();
  return result;
}


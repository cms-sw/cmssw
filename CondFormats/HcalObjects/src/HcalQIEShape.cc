/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store pedestal values 4xCapId
$Author: ratnikov
$Date: 2013/03/25 16:23:33 $
$Revision: 1.8 $
*/
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"

HcalQIEShape::HcalQIEShape()
  :nbins_(0)
{
}

HcalQIEShape::~HcalQIEShape() {}

void HcalQIEShape::expand () {
  int scale = 1;
  for (unsigned range = 1; range < 4; range++) {
    int factor = nbins_ == 32 ? 5 : 8;  // QIE8/QIE10 -> 5/8
    scale *= factor;
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
  mValues.resize(4*nbins_+1);
  bool result = true;
  for (unsigned int adc = 0; adc < nbins_; adc++) result = result && setLowEdge (fValue [adc], adc);
  expand ();
  return result;
}


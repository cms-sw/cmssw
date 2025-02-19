/** \class HcalChannelCoder
    
    Container for ADC<->fQ conversion constants for HCAL QIE
   $Author: ratnikov
   $Revision: 1.3 $
*/

#include <iostream>

#include "CalibFormats/HcalObjects/interface/HcalChannelCoder.h"
#include "CalibFormats/HcalObjects/interface/QieShape.h"

HcalChannelCoder::HcalChannelCoder (const float fOffset [16], const float fSlope [16]) { // [CapId][Range]
  for (int range = 0; range < 4; range++) {
    for (int capId = 0; capId < 4; capId++) {
      mOffset [capId][range] = fOffset [index (capId, range)];
      mSlope [capId][range] = fSlope [index (capId, range)];
    }
  }
}

double HcalChannelCoder::charge (const QieShape& fShape, int fAdc, int fCapId) const {
  int range = (fAdc >> 6) & 0x3;
  double charge = fShape.linearization (fAdc) / mSlope [fCapId][range] + mOffset [fCapId][range];
//   std::cout << "HcalChannelCoder::charge-> " << fAdc << '/' << fCapId 
// 	    << " result: " << charge << std::endl;
  return charge;
}

int HcalChannelCoder::adc (const QieShape& fShape, double fCharge, int fCapId) const {

  int adc = -1; //nothing found yet
  // search for the range
  for (int range = 0; range < 4; range++) {
    double qieCharge = (fCharge - mOffset [fCapId][range]) * mSlope [fCapId][range];
    double qieChargeMax = fShape.linearization (32*range+31) + 0.5 * fShape.binSize (32*range+31);
    if (range == 3 && qieCharge > qieChargeMax) adc = 127; // overflow
    if (qieCharge > qieChargeMax) continue; // next range
    for (int bin = 32*range; bin < 32*(range+1); bin++) {
      if (qieCharge < fShape.linearization (bin) + 0.5 * fShape.binSize (bin)) {
        adc = bin;
        break;
      }
    }
    if (adc >= 0) break; // found
  }
  if (adc < 0) adc = 0; // underflow

//   std::cout << "HcalChannelCoder::adc-> " << fCharge << '/' << fCapId 
// 	    << " result: " << adc << std::endl;
  return adc;
}


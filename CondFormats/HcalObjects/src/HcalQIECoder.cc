/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store QIE coder parameters for one channel
$Author: ratnikov
$Date: 2005/11/07 22:15:09 $
$Revision: 1.3 $
*/

#include <iostream>

#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"

namespace {
  // pack range/capId in the plain index
  unsigned index (unsigned fRange, unsigned fCapId) {return fCapId * 4 + fRange;}
  unsigned range (unsigned fIndex) {return fIndex % 4;}
  unsigned capId (unsigned fIndex) {return fIndex / 4;}
}

float HcalQIECoder::charge (const HcalQIEShape& fShape, unsigned fAdc, unsigned fCapId) const {
  unsigned range = fShape.range (fAdc);
  return (fShape.center (fAdc) - offset (fCapId, range)) / slope (fCapId, range);
}

unsigned HcalQIECoder::adc (const HcalQIEShape& fShape, float fCharge, unsigned fCapId) const {
  unsigned adc = 0; //nothing found yet
  // search for the range
  for (unsigned range = 0; range < 4; range++) {
    float qieCharge = fCharge * slope (fCapId, range) + offset (fCapId, range);
    unsigned minBin = 32*range;
    float qieChargeMax = fShape.highEdge (minBin + 31);
    if (range == 3 && qieCharge > qieChargeMax) {
      adc = 127; // overflow
      break;
    }
    if (qieCharge > qieChargeMax) continue; // next range
    for (unsigned bin = minBin; bin <= minBin + 31; bin++) {
      if (qieCharge < fShape.highEdge (bin)) {
        adc = bin;
        break;
      }
    }
  }
  return adc;
}

float HcalQIECoder::offset (unsigned fCapId, unsigned fRange) const {
  return *((&mOffset00) + index (fRange, fCapId));
}

float HcalQIECoder::slope (unsigned fCapId, unsigned fRange) const {
  return *((&mSlope00) + index (fRange, fCapId));
}
  
void HcalQIECoder::setOffset (unsigned fCapId, unsigned fRange, float fValue) {
  if (fCapId >= 0 && fRange >= 0 && fCapId < 4 && fRange < 4) {
    *((&mOffset00) + index (fRange, fCapId)) = fValue;
  }
  else {
    std::cerr << "HcalQIECoder::setOffset-> Wrong parameters capid/range: " << fCapId << '/' << fRange << std::endl;
  }
}

void HcalQIECoder::setSlope (unsigned fCapId, unsigned fRange, float fValue) {
  if (fCapId >= 0 && fRange >= 0 && fCapId < 4 && fRange < 4) {
    *((&mSlope00) + index (fRange, fCapId)) = fValue;
  }
  else {
    std::cerr << "HcalQIECoder::setSlope-> Wrong parameters capid/range: " << fCapId << '/' << fRange << std::endl;
  }
}


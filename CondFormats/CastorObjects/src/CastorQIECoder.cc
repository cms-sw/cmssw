/** 
\class CastorQIEData
\author Panos Katsas (UoA)
POOL object to store QIE coder parameters for one channel
*/

#include <iostream>

#include "CondFormats/CastorObjects/interface/CastorQIEShape.h"
#include "CondFormats/CastorObjects/interface/CastorQIECoder.h"

namespace {
  // pack range/capId in the plain index
  unsigned index (unsigned fRange, unsigned fCapId) {return fCapId * 4 + fRange;}
  inline unsigned range (unsigned fIndex) {return fIndex % 4;}
  inline unsigned capId (unsigned fIndex) {return fIndex / 4;}
}

float CastorQIECoder::charge (const CastorQIEShape& fShape, unsigned fAdc, unsigned fCapId) const {
  unsigned range = fShape.range (fAdc);
  return (fShape.center (fAdc) - offset (fCapId, range)) / slope (fCapId, range);
}

unsigned CastorQIECoder::adc (const CastorQIEShape& fShape, float fCharge, unsigned fCapId) const {
  // search for the range
  for (unsigned range = 0; range < 4; range++) {
    float qieCharge = fCharge * slope (fCapId, range) + offset (fCapId, range);
    unsigned minBin = 32*range;
    float qieChargeMax = fShape.highEdge (minBin + 31);
    if (qieCharge <= qieChargeMax) {
      for (unsigned bin = minBin; bin <= minBin + 31; bin++) {
	if (qieCharge < fShape.highEdge (bin)) {
	  return bin;
	}
      }
      return minBin; // underflow
    }
    else if (range == 3) {
      return 127; // overflow
    }
  }
  return 0; //should never get here
}

float CastorQIECoder::offset (unsigned fCapId, unsigned fRange) const {
  return *((&mOffset00) + index (fRange, fCapId));
}

float CastorQIECoder::slope (unsigned fCapId, unsigned fRange) const {
  return *((&mSlope00) + index (fRange, fCapId));
}
  
void CastorQIECoder::setOffset (unsigned fCapId, unsigned fRange, float fValue) {
  if (fCapId < 4U && fRange < 4U) { // fCapId >= 0 and fRange >= 0, since fCapId and fRange are unsigned
     *((&mOffset00) + index (fRange, fCapId)) = fValue;
  }
  else {
    std::cerr << "CastorQIECoder::setOffset-> Wrong parameters capid/range: " << fCapId << '/' << fRange << std::endl;
  }
}

void CastorQIECoder::setSlope (unsigned fCapId, unsigned fRange, float fValue) {
  if (fCapId < 4U && fRange < 4U) { // fCapId >= 0 and fRange >= 0, since fCapId and fRange are unsigned
    *((&mSlope00) + index (fRange, fCapId)) = fValue;
  }
  else {
    std::cerr << "CastorQIECoder::setSlope-> Wrong parameters capid/range: " << fCapId << '/' << fRange << std::endl;
  }
}


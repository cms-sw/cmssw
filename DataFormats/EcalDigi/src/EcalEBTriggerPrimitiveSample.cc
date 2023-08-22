#include "DataFormats/EcalDigi/interface/EcalEBTriggerPrimitiveSample.h"
#include <iostream>

EcalEBTriggerPrimitiveSample::EcalEBTriggerPrimitiveSample() : theSample(0) {}
EcalEBTriggerPrimitiveSample::EcalEBTriggerPrimitiveSample(uint32_t data) : theSample(data) {
  theSample = theSample & 0x3ffff;
}

EcalEBTriggerPrimitiveSample::EcalEBTriggerPrimitiveSample(int encodedEt, bool isASpike) {
  theSample = (encodedEt & 0xFFF) | ((isASpike) ? (0x1000) : (0));
  theSample = theSample & 0x3ffff;
}

EcalEBTriggerPrimitiveSample::EcalEBTriggerPrimitiveSample(int encodedEt, bool isASpike, int timing) {
  theSample = (encodedEt & 0xFFF) | ((isASpike) ? (0x1000) : (0)) | timing << 13;
  theSample = theSample & 0x3ffff;

  //  std::cout << " EcalEBTriggerPrimitiveSample encodedEt "<< encodedEt << " isASpike " << isASpike << " time " << timing << " timing <<13 " << timing<<13 << " theSample "<< theSample << std::endl;
}

EcalEBTriggerPrimitiveSample::EcalEBTriggerPrimitiveSample(int encodedEt) {
  theSample = encodedEt & 0xFFF;
  theSample = theSample & 0x3ffff;
}

std::ostream& operator<<(std::ostream& s, const EcalEBTriggerPrimitiveSample& samp) {
  return s << "ET=" << samp.encodedEt() << ", isASpike=" << samp.l1aSpike() << " timing= " << samp.time();
}

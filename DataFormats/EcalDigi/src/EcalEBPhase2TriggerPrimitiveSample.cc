#include "DataFormats/EcalDigi/interface/EcalEBPhase2TriggerPrimitiveSample.h"
#include <iostream>

EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample() : theSample(0) {}
EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample(uint32_t data) : theSample(data) {
  theSample = theSample & 0x3ffff;
}

EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample(int encodedEt, bool isASpike) {
  theSample = (encodedEt & 0xFFF) | ((isASpike) ? (0x1000) : (0));
  theSample = theSample & 0x3ffff;
}

EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample(int encodedEt, bool isASpike, int timing) {
  theSample = (encodedEt & 0xFFF) | ((isASpike) ? (0x1000) : (0)) | timing << 13;
  theSample = theSample & 0x3ffff;

  //  std::cout << " EcalEBPhase2TriggerPrimitiveSample encodedEt "<< encodedEt << " isASpike " << isASpike << " time " << timing << " timing <<13 " << timing<<13 << " theSample "<< theSample << std::endl;
}

EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample(int encodedEt) {
  theSample = encodedEt & 0xFFF;
  theSample = theSample & 0x3ffff;
}

std::ostream& operator<<(std::ostream& s, const EcalEBPhase2TriggerPrimitiveSample& samp) {
  return s << "ET=" << samp.encodedEt() << ", isASpike=" << samp.l1aSpike() << " timing= " << samp.time();
}

#include "DataFormats/EcalDigi/interface/EcalEBTriggerPrimitiveSample.h"
#include <iostream>

EcalEBTriggerPrimitiveSample::EcalEBTriggerPrimitiveSample() : theSample(0) {}
EcalEBTriggerPrimitiveSample::EcalEBTriggerPrimitiveSample(uint16_t data) : theSample(data) {}

EcalEBTriggerPrimitiveSample::EcalEBTriggerPrimitiveSample(int encodedEt, bool isASpike) {
  theSample = (encodedEt & 0x3FF) | ((isASpike) ? (0x400) : (0));
}

EcalEBTriggerPrimitiveSample::EcalEBTriggerPrimitiveSample(int encodedEt, bool isASpike, int timing) {
  theSample = (encodedEt & 0x3FF) | ((isASpike) ? (0x400) : (0)) | timing << 11;
}

EcalEBTriggerPrimitiveSample::EcalEBTriggerPrimitiveSample(int encodedEt) { theSample = encodedEt & 0x3FF; }

std::ostream& operator<<(std::ostream& s, const EcalEBTriggerPrimitiveSample& samp) {
  return s << "ET=" << samp.encodedEt() << ", isASpike=" << samp.l1aSpike() << " timing= " << samp.time();
}

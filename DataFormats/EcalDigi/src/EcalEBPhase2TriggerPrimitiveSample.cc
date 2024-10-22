#include "DataFormats/EcalDigi/interface/EcalEBPhase2TriggerPrimitiveSample.h"
#include <iostream>

EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample() : theSample_(0) {}
EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample(uint32_t data) : theSample_(data) {
  theSample_ = theSample_ & 0x3ffff;
}

EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample(int encodedEt, bool isASpike) {
  theSample_ = (encodedEt & 0xFFF) | ((isASpike) ? (0x1000) : (0));
  theSample_ = theSample_ & 0x3ffff;
}

EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample(int encodedEt, bool isASpike, int timing) {
  theSample_ = (encodedEt & 0xFFF) | ((isASpike) ? (0x1000) : (0)) | timing << 13;
  theSample_ = theSample_ & 0x3ffff;
}

EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample(int encodedEt) {
  theSample_ = encodedEt & 0xFFF;
  theSample_ = theSample_ & 0x3ffff;
}

std::ostream& operator<<(std::ostream& s, const EcalEBPhase2TriggerPrimitiveSample& samp) {
  return s << "ET=" << samp.encodedEt() << ", isASpike=" << samp.l1aSpike() << " timing= " << samp.time();
}

#include "DataFormats/EcalDigi/interface/EcalEBPhase2TriggerPrimitiveSample.h"
#include <iostream>

EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample() : theSample_(0) {}
EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample(uint32_t data) : theSample_(data) {
  theSample_ = raw();
}

EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample(int encodedEt, bool isASpike) {
  theSample_ = (encodedEt & 0x3FF) | ((isASpike) ? (0x400) : (0));
  theSample_ = raw();
}

EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample(int encodedEt, bool isASpike, int timing) {
  theSample_ = (encodedEt & 0x3FF) | ((isASpike) ? (0x400) : (0)) | timing << 11;
  theSample_ = raw();
}

EcalEBPhase2TriggerPrimitiveSample::EcalEBPhase2TriggerPrimitiveSample(int encodedEt) {
  theSample_ = encodedEt & 0x3FF;
  theSample_ = raw();
}

std::ostream& operator<<(std::ostream& s, const EcalEBPhase2TriggerPrimitiveSample& samp) {
  return s << "ET=" << samp.encodedEt() << ", isASpike=" << samp.l1aSpike() << " timing= " << samp.time();
}

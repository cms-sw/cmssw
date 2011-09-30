#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"



EcalTriggerPrimitiveSample::EcalTriggerPrimitiveSample() : theSample(0) { }
EcalTriggerPrimitiveSample::EcalTriggerPrimitiveSample(uint16_t data) : theSample(data) { }

EcalTriggerPrimitiveSample::EcalTriggerPrimitiveSample(int encodedEt, bool fineGrain, int ttFlag) { 
  theSample=((ttFlag&0x7)<<9)|(encodedEt&0xFF)|
    ((fineGrain)?(0x100):(0));
}


std::ostream& operator<<(std::ostream& s, const EcalTriggerPrimitiveSample& samp) {
  return s << "ET=" << samp.compressedEt() << ", FG=" << samp.fineGrain() << ", TTF=" << samp.ttFlag();
}



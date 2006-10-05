#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"


HcalTriggerPrimitiveSample::HcalTriggerPrimitiveSample() : theSample(0) { }
HcalTriggerPrimitiveSample::HcalTriggerPrimitiveSample(uint16_t data) : theSample(data) { }

HcalTriggerPrimitiveSample::HcalTriggerPrimitiveSample(int encodedEt, bool fineGrain, int fiber, int fiberchan) { 
  theSample=(((fiber-1)&0x7)<<13)|((fiberchan&0x3)<<11)|(encodedEt&0xFF)|
    ((fineGrain)?(0x100):(0));
}


std::ostream& operator<<(std::ostream& s, const HcalTriggerPrimitiveSample& samp) {
  return s << "Value=" << samp.compressedEt() << ", FG=" << samp.fineGrain();
}



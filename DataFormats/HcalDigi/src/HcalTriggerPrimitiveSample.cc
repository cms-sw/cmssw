#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"

HcalTriggerPrimitiveSample::HcalTriggerPrimitiveSample() : theSample(0) {}
HcalTriggerPrimitiveSample::HcalTriggerPrimitiveSample(uint16_t data) : theSample(data) {}

HcalTriggerPrimitiveSample::HcalTriggerPrimitiveSample(int encodedEt, bool fineGrain, int slb, int slbchan) {
  theSample = (((slb)&0x7) << 13) | ((slbchan & 0x3) << 11) | (encodedEt & 0xFF) | ((fineGrain) ? (0x100) : (0));
}

HcalTriggerPrimitiveSample::HcalTriggerPrimitiveSample(int encodedEt, int fineGrainExt) {
  theSample = (encodedEt & 0xFF) | ((fineGrainExt & 0x3F) << 8);
}

std::ostream& operator<<(std::ostream& s, const HcalTriggerPrimitiveSample& samp) {
  return s << "Value=" << samp.compressedEt() << ", FG=" << samp.fineGrain();
}

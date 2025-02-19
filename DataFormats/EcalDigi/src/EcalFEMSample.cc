#include "DataFormats/EcalDigi/interface/EcalFEMSample.h"

EcalFEMSample::EcalFEMSample(int adc, int gainId) {
  theSample=(adc&0xFFF) | ((gainId&0x3)<<12);
}

std::ostream& operator<<(std::ostream& s, const EcalFEMSample& samp) {
  s << "ADC=" << samp.adc() << ", gainId=" << samp.gainId();
  return s;
}

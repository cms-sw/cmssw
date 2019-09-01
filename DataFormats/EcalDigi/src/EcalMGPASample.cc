#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include <iostream>

EcalMGPASample::EcalMGPASample(int adc, int gainId) { theSample = (adc & 0xFFF) | ((gainId & 0x3) << 12); }

std::ostream& operator<<(std::ostream& s, const EcalMGPASample& samp) {
  s << "ADC=" << samp.adc() << ", gainId=" << samp.gainId();
  return s;
}

#include "DataFormats/EcalDigi/interface/EcalLiteDTUSample.h"
#include <iostream>

EcalLiteDTUSample::EcalLiteDTUSample(int adc, int gainId) {
  theSample = (adc & ecalPh2::kAdcMask) | ((gainId & ecalPh2::kGainIdMask) << ecalPh2::NBITS);
}

std::ostream& operator<<(std::ostream& s, const EcalLiteDTUSample& samp) {
  s << "ADC=" << samp.adc() << ", gainId=" << samp.gainId();
  return s;
}

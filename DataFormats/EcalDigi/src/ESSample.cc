#include "DataFormats/EcalDigi/interface/ESSample.h"

ESSample::ESSample(const int& adc) {
  theSample = (adc&0xFFF);
}

std::ostream& operator<<(std::ostream& s, const ESSample& samp) {
  s << "ADC = " << samp.adc() ;
  return s;
}

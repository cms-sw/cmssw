#include "DataFormats/EcalDigi/interface/EcalMatacqSample.h"

EcalMatacqSample::EcalMatacqSample(int adc) {
  theSample=(adc&0xFFFF) ;
}

std::ostream& operator<<(std::ostream& s, const EcalMatacqSample& samp) {
  s << "ADC=" << samp.adc() ;
  return s;
}

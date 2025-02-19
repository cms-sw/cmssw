#include "DataFormats/EcalDigi/interface/ESSample.h"

ESSample::ESSample(int adc) {
  theSample = (int16_t)adc;
}

std::ostream& operator<<(std::ostream& s, const ESSample& samp) {
  s << "ADC = " << samp.adc() ;
  return s;
}

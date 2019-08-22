#include "DataFormats/HcalDigi/interface/HcalQIESample.h"

std::ostream& operator<<(std::ostream& s, const HcalQIESample& samp) {
  s << "ADC=" << samp.adc() << ", capid=" << samp.capid();
  if (samp.er())
    s << ", ER";
  if (samp.dv())
    s << ", DV";
  return s;
}

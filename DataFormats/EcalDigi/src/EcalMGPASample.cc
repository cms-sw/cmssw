#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"

namespace cms {

  EcalMGPASample::EcalMGPASample(int adc, int gainId) {
    theSample=(adc&0xFF) | ((gainId&0x3)<<12);
  }

  std::ostream& operator<<(std::ostream& s, const EcalMGPASample& samp) {
    s << "ADC=" << samp.adc() << ", capid=" << samp.gainId();
    return s;
  }

}

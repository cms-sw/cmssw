#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include <ostream>
std::ostream & operator<<(std::ostream & o, const SiStripDigi& digi) {
  return o << " " << digi.strip()
           << " " << digi.adc();
}


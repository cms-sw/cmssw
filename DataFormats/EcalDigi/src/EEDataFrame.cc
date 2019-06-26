#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include <iostream>

std::ostream& operator<<(std::ostream& s, const EEDataFrame& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i = 0; i < digi.size(); i++)
    s << "  " << digi.sample(i) << std::endl;
  return s;
}

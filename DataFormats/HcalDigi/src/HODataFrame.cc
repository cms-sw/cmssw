#include "DataFormats/HcalDigi/interface/HODataFrame.h"

std::ostream& operator<<(std::ostream& s, const HODataFrame& digi) {
  s << digi.id() << " " << digi.size() << " samples  " << digi.presamples() << " presamples ";
  if (digi.zsUnsuppressed())
    s << " zsUS ";
  if (digi.zsMarkAndPass())
    s << " zsM&P ";
  if (digi.fiberIdleOffset() != 0) {
    if (digi.fiberIdleOffset() == -1000)
      s << " nofiberOffset";
    else
      s << " fiberOffset=" << digi.fiberIdleOffset();
  }
  s << std::endl;
  for (int i = 0; i < digi.size(); i++)
    s << "  " << digi.sample(i) << std::endl;
  return s;
}

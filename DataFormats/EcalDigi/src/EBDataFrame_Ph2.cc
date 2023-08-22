#include "DataFormats/EcalDigi/interface/EBDataFrame_Ph2.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

std::ostream& operator<<(std::ostream& s, const EBDataFrame_Ph2& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i = 0; i < digi.size(); i++)
    s << "  " << digi[i] << std::endl;
  return s;
}


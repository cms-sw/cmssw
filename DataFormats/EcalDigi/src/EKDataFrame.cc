#include "DataFormats/EcalDigi/interface/EKDataFrame.h"
#include<iostream>

std::ostream& operator<<(std::ostream& s, const EKDataFrame& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi.sample(i) << std::endl;
  return s;
}


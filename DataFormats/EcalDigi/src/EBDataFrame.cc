#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include<iostream>


std::ostream& operator<<(std::ostream& s, const EBDataFrame& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi[i] << std::endl;
  return s;
}

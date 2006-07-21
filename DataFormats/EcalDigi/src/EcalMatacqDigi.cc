#include "DataFormats/EcalDigi/interface/EcalMatacqDigi.h"


EcalMatacqDigi::EcalMatacqDigi() : size_(0), data_(MAXSAMPLES) {
}
  
void EcalMatacqDigi::setSize(int size) {
  if (size<0) size_=0;
  else if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else size_=size;
}

  
std::ostream& operator<<(std::ostream& s, const EcalMatacqDigi& digi) {
  s << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi.sample(i) << std::endl;
  return s;
}

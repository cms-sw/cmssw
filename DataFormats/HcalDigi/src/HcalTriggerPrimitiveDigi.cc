#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"


HcalTriggerPrimitiveDigi::HcalTriggerPrimitiveDigi() : size_(0), hcalPresamples_(0),data_(MAXSAMPLES) {
}
HcalTriggerPrimitiveDigi::HcalTriggerPrimitiveDigi(const HcalTrigTowerDetId& id) : id_(id),
										   size_(0), hcalPresamples_(0),data_(MAXSAMPLES) {
}
  
void HcalTriggerPrimitiveDigi::setSize(int size) {
  if (size<0) size_=0;
  else if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else size_=size;
}
void HcalTriggerPrimitiveDigi::setPresamples(int ps) {
  if (ps<0) hcalPresamples_=0;
  //  else if (ps>=size_) hcalPresamples_=size_-1;
  else hcalPresamples_=ps;
}
  
std::ostream& operator<<(std::ostream& s, const HcalTriggerPrimitiveDigi& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi.sample(i) << std::endl;
  return s;
}
  


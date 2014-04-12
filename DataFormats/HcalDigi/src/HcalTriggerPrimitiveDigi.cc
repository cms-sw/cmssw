#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"


HcalTriggerPrimitiveDigi::HcalTriggerPrimitiveDigi() : size_(0), hcalPresamples_(0) {
}
HcalTriggerPrimitiveDigi::HcalTriggerPrimitiveDigi(const HcalTrigTowerDetId& id) : id_(id),
										   size_(0), hcalPresamples_(0) {
}
  
void HcalTriggerPrimitiveDigi::setSize(int size) {
  if (size<0) size_=0;
  else if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else size_=size;
}
void HcalTriggerPrimitiveDigi::setPresamples(int ps) {
  if (ps<0) hcalPresamples_&=0xFFFFFF0;
  //  else if (ps>=size_) hcalPresamples_=size_-1;
  else hcalPresamples_|=ps&0xF;
}

void HcalTriggerPrimitiveDigi::setZSInfo(bool unsuppressed, bool markAndPass) {
  if (markAndPass) hcalPresamples_|=0x10;
  if (unsuppressed) hcalPresamples_|=0x20;
}

std::ostream& operator<<(std::ostream& s, const HcalTriggerPrimitiveDigi& digi) {
  s << digi.id() << " " << digi.size() << " samples "<< digi.presamples() << " presamples";
  if (digi.zsUnsuppressed()) s << " zsUS";
  if (digi.zsMarkAndPass()) s << " zsM&P";
  s << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi.sample(i) << std::endl;
  return s;
}
  


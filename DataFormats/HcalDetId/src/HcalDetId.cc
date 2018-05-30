#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <ostream>
#include <iostream>

const HcalDetId HcalDetId::Undefined(HcalEmpty,0,0,0);

std::ostream& operator<<(std::ostream& s,const HcalDetId& id) {
  switch (id.subdet()) {
  case(HcalBarrel) : return s << "(HB " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(HcalEndcap) : return s << "(HE " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(HcalForward) : return s << "(HF " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(HcalOuter) : return s << "(HO " << id.ieta() << ',' << id.iphi() << ')';
  case(HcalTriggerTower) : return s << "(HT " << id.ieta() << ',' << id.iphi() << ')';
  default : return s << std::hex << id.rawId() << std::dec;
  }
}



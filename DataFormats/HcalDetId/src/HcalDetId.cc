#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

const HcalDetId HcalDetId::Undefined(HcalEmpty,0,0,0);

HcalDetId::HcalDetId() : DetId() {
}

HcalDetId::HcalDetId(uint32_t rawid) : DetId(rawid) {
}

HcalDetId::HcalDetId(HcalSubdetector subdet, int tower_ieta, int tower_iphi, int depth) : DetId(Hcal,subdet) {
  // (no checking at this point!)
  id_ |= ((depth&0x7)<<14) |
    ((tower_ieta>0)?(0x2000|(tower_ieta<<7)):((-tower_ieta)<<7)) |
    (tower_iphi&0x7F);
}

HcalDetId::HcalDetId(const DetId& gen) {
  if (!gen.null()) {
    HcalSubdetector subdet=(HcalSubdetector(gen.subdetId()));
    if (gen.det()!=Hcal || 
	(subdet!=HcalBarrel && subdet!=HcalEndcap && 
	 subdet!=HcalOuter && subdet!=HcalForward ))
      {
	throw cms::Exception("Invalid DetId") << "Cannot initialize HcalDetId from " << std::hex << gen.rawId() << std::dec; 
      }  
  }
  id_=gen.rawId();
}

HcalDetId& HcalDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    HcalSubdetector subdet=(HcalSubdetector(gen.subdetId()));
    if (gen.det()!=Hcal || 
	(subdet!=HcalBarrel && subdet!=HcalEndcap && 
	 subdet!=HcalOuter && subdet!=HcalForward ))
      {
	throw cms::Exception("Invalid DetId") << "Cannot assign HcalDetId from " << std::hex << gen.rawId() << std::dec; 
      }  
  }
  id_=gen.rawId();
  return (*this);
}

int HcalDetId::crystal_iphi_low() const { 
  int simple_iphi=((iphi()-1)*5)+1; 
  simple_iphi+=10;
  return ((simple_iphi>360)?(simple_iphi-360):(simple_iphi));
}

int HcalDetId::crystal_iphi_high() const { 
  int simple_iphi=((iphi()-1)*5)+5; 
  simple_iphi+=10;
  return ((simple_iphi>360)?(simple_iphi-360):(simple_iphi));
}

std::ostream& operator<<(std::ostream& s,const HcalDetId& id) {
  switch (id.subdet()) {
  case(HcalBarrel) : return s << "(HB " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(HcalEndcap) : return s << "(HE " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(HcalForward) : return s << "(HF " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(HcalOuter) : return s << "(HO " << id.ieta() << ',' << id.iphi() << ')';
  default : return s << id.rawId();
  }
}

int HcalDetId::hashed_index() const {
  int index = -1;

  int HBhalf = 1296;
  int HEhalf = 1296;
  int HOhalf = 1080;
  int HFhalf = 864;

  // HB valid DetIds: phi=1-72,eta=1-14,depth=1; phi=1-72,eta=15-16,depth=1-2
  if (subdet() == HcalBarrel)
    {
      if (ietaAbs() < 16)   index = (iphi() - 1)*18 + (ietaAbs() - 1) + (depth() - 1);
      if (ietaAbs() == 16)  index = (iphi() - 1)*18 + ietaAbs() + (depth() - 1);
      
      if (zside() == -1) index += HBhalf;
    }
  
  // HE valid DetIds: phi=1-72,eta=16-17,depth=1; phi=1-72,eta=18-20,depth=1-2; 
  //                  phi=1-71(in steps of 2),eta=21-26,depth=1-2; phi=1-71(in steps of 2),eta=27-28,depth=1-3
  //                  phi=1-71(in steps of 2),eta=29,depth=1-2
  if (subdet() == HcalEndcap)
    {
      if (ietaAbs() == 16 || ietaAbs() == 17)  index = (iphi() - 1)*8 + (iphi()/2)*20 + (ietaAbs() - 16);
      if (ietaAbs() >= 18 && ietaAbs() <= 20)  index = (iphi() - 1)*8 + (iphi()/2)*20 + 2  + 2*(ietaAbs()-18) + (depth() - 1);
      if (ietaAbs() >= 21 && ietaAbs() <= 26)  index = (iphi() - 1)*8 + (iphi()/2)*20 + 8  + 2*(ietaAbs()-21) + (depth() - 1);
      if (ietaAbs() >= 27 && ietaAbs() <= 28)  index = (iphi() - 1)*8 + (iphi()/2)*20 + 20 + 3*(ietaAbs()-27) + (depth() - 1);
      if (ietaAbs() == 29)                     index = (iphi() - 1)*8 + (iphi()/2)*20 + 26 + 2*(ietaAbs()-29) + (depth() - 1);
      
      index += 2*HBhalf;
      if (zside() == -1) index += HEhalf;
    }

  // HO valid DetIds: phi=1-72,eta=1-15,depth=4!
  if (subdet() == HcalOuter)
    {
      index = (2*HBhalf) + (2*HEhalf) + (iphi() - 1)*15 + (ietaAbs() - 1);

      if (zside() == -1) index += HOhalf;
  }

  // HF valid DetIds: phi=1-71(in steps of 2),eta=29-39,depth=1-2; phi=3-71(in steps of 4),eta=40-41,depth=1-2
  if (subdet() == HcalForward)
    {
      index = ((iphi()-1)/4)*4 + ((iphi()-1)/2)*22 + 2*(ietaAbs()-29) + (depth() - 1);

      index += (2*HBhalf) + (2*HEhalf) + (2*HOhalf);
      if (zside() == -1)  index += HFhalf;
    }
  return index;
}

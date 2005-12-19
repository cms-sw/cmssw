#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"



EcalTrigTowerDetId::EcalTrigTowerDetId() {
}
  
  
EcalTrigTowerDetId::EcalTrigTowerDetId(uint32_t rawid) : DetId(rawid) {
}
  
EcalTrigTowerDetId::EcalTrigTowerDetId(int ieta, int iphi) : DetId(Ecal,EcalTriggerTower) {
  id_|=((ieta>0)?(0x2000|(ieta<<7)):((-ieta)<<7)) |
    (iphi&0x7F);
}
  
EcalTrigTowerDetId::EcalTrigTowerDetId(const DetId& gen) {
  if (gen.det()!=Ecal || gen.subdetId()!=EcalTriggerTower) {
    throw new std::exception();
  }
  id_=gen.rawId();
}
  
EcalTrigTowerDetId& EcalTrigTowerDetId::operator=(const DetId& gen) {
  if (gen.det()!=Ecal || gen.subdetId()!=EcalTriggerTower) {
    throw new std::exception();
  }
  id_=gen.rawId();
  return *this;
}

//New SM numbering scheme. Avoids discontinuity in phi crossing \eta=0  
int EcalTrigTowerDetId::iDCC() const {
  int id = ( iphi() - 1 ) / kTowersInPhi + 1;
  if ( zside() < 0 ) id += 18;
  return id;
}

int
EcalTrigTowerDetId::iTT() const {
  int ie = ietaAbs() -1;
  int ip;
  if (zside() < 0) {
    ip = (( iphi() -1 ) % kTowersInPhi ) + 1;
  } else {
    ip = kTowersInPhi - ((iphi() -1 ) % kTowersInPhi );
  }

  return (ie * kTowersInPhi) + ip;
}

int
EcalTrigTowerDetId::hashedIndex() const {

  return (iDCC()-1) * kTowersPerSM + iTT() - 1;

}


std::ostream& operator<<(std::ostream& s,const EcalTrigTowerDetId& id) {
  return s << "(EcalTrigTower " << id.ieta() << ',' << id.iphi() << ')';
}


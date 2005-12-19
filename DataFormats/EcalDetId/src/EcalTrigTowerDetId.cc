#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"



EcalTrigTowerDetId::EcalTrigTowerDetId() {
}
  
  
EcalTrigTowerDetId::EcalTrigTowerDetId(uint32_t rawid) : DetId(rawid) {
}

EcalTrigTowerDetId::EcalTrigTowerDetId(int zside, EcalSubdetector subDet, int i, int j, int mode) throw(std::runtime_error) 
  : DetId(Ecal,EcalTriggerTower) 
{
  int tower_i=0;
  int tower_j=0;

  if (mode == SUBDETIJMODE)
    {
      tower_i=i;
      tower_j=j;
    }
  else if (mode == SUBDETDCCTTMODE)
    {
      throw(std::runtime_error("EcalTriggerTowerDetId:  Cannot create object. SUBDETDCCTTMODE not yet implemented."));   
    }
  else
    throw(std::runtime_error("EcalTriggerTowerDetId:  Cannot create object.  Unknown mode for (int, EcalSubdetector, int, int) constructor."));
  
  if (tower_i > MAX_I || tower_i < MIN_I  || tower_j > MAX_J || tower_j < MIN_J)
    throw(std::runtime_error("EcalTriggerTowerDetId:  Cannot create object.  Indexes out of bounds."));
  
  id_|= ((zside>0)?(0x8000):(0x0)) | ((subDet == EcalBarrel)?(0x4000):(0x0)) | (tower_i<<7) | (tower_j & 0x7F);

}
  
EcalTrigTowerDetId::EcalTrigTowerDetId(const DetId& gen) 
{
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
int EcalTrigTowerDetId::iDCC() const throw(std::runtime_error)
{
  if ( subDet() == EcalBarrel )
    {
      int id = ( iphi() - 1 ) / kTowersInPhi + 1;
      if ( zside() < 0 ) id += 18;
      return id;
    }
  else
    throw(std::runtime_error("EcalTriggerTowerDetId: iDCC not yet implemented"));
}

int EcalTrigTowerDetId::iTT() const throw(std::runtime_error)
{
  if ( subDet() == EcalBarrel )
    {
      int ie = ietaAbs() -1;
      int ip;
      if (zside() < 0) {
	ip = (( iphi() -1 ) % kTowersInPhi ) + 1;
      } else {
	ip = kTowersInPhi - ((iphi() -1 ) % kTowersInPhi );
      }
      
      return (ie * kTowersInPhi) + ip;
    }
  else
    throw(std::runtime_error("EcalTriggerTowerDetId: iTT not yet implemented"));
}

int EcalTrigTowerDetId::hashedIndex() const 
{
  return (iDCC()-1) * kTowersPerSM + iTT() - 1;
}


std::ostream& operator<<(std::ostream& s,const EcalTrigTowerDetId& id) {
  return s << "(EcalTrigTower " << id.ieta() << ',' << id.iphi() << ')';
}


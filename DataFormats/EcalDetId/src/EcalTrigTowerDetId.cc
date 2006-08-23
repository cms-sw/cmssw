#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "FWCore/Utilities/interface/Exception.h"


EcalTrigTowerDetId::EcalTrigTowerDetId() {
}
  
  
EcalTrigTowerDetId::EcalTrigTowerDetId(uint32_t rawid) : DetId(rawid) {
}

EcalTrigTowerDetId::EcalTrigTowerDetId(int zside, EcalSubdetector subDet, int i, int j, int mode)
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
      throw cms::Exception("InvalidDetId") << "EcalTriggerTowerDetId:  Cannot create object. SUBDETDCCTTMODE not yet implemented.";   
    }
  else
    throw cms::Exception("InvalidDetId") << "EcalTriggerTowerDetId:  Cannot create object.  Unknown mode for (int, EcalSubdetector, int, int) constructor.";
  
  if (tower_i > MAX_I || tower_i < MIN_I  || tower_j > MAX_J || tower_j < MIN_J)
    throw cms::Exception("InvalidDetId") << "EcalTriggerTowerDetId:  Cannot create object.  Indexes out of bounds.";
  
  id_|= ((zside>0)?(0x8000):(0x0)) | ((subDet == EcalBarrel)?(0x4000):(0x0)) | (tower_i<<7) | (tower_j & 0x7F);

}
  
EcalTrigTowerDetId::EcalTrigTowerDetId(const DetId& gen) 
{
  if (!gen.null() && ( gen.det()!=Ecal || gen.subdetId()!=EcalTriggerTower )) {
    throw cms::Exception("InvalidDetId");  }
  id_=gen.rawId();
}
  
EcalTrigTowerDetId& EcalTrigTowerDetId::operator=(const DetId& gen) {
  if (!gen.null() && ( gen.det()!=Ecal || gen.subdetId()!=EcalTriggerTower )) {
    throw cms::Exception("InvalidDetId");
  }
  id_=gen.rawId();
  return *this;
}

//New SM numbering scheme. Avoids discontinuity in phi crossing \eta=0  
int EcalTrigTowerDetId::iDCC() const 
{
  if ( subDet() == EcalBarrel )
    {
      int id = ( iphi() - 1 ) / kEBTowersInPhi + 1;
      if ( zside() < 0 ) id += 18;
      return id;
    }
  else
    throw cms::Exception("MethodNotImplemented") << "EcalTriggerTowerDetId: iDCC not yet implemented";
}

int EcalTrigTowerDetId::iTT() const 
{
  if ( subDet() == EcalBarrel )
    {
      int ie = ietaAbs() -1;
      int ip;
      if (zside() < 0) {
	ip = (( iphi() -1 ) % kEBTowersInPhi ) + 1;
      } else {
	ip = kEBTowersInPhi - ((iphi() -1 ) % kEBTowersInPhi );
      }
      
      return (ie * kEBTowersInPhi) + ip;
    }
  else
    throw cms::Exception("MethodNotImplemented") << "EcalTriggerTowerDetId: iTT not yet implemented";
}

int EcalTrigTowerDetId::iquadrant() const
{
  if ( subDet() == EcalEndcap )
    return int((iphi()-1)/kEETowersInPhiPerQuadrant)+1;
  else
    throw cms::Exception("MethodNotApplicable") << "EcalTriggerTowerDetId: iquadrant not applicable";
}  

int EcalTrigTowerDetId::hashedIndex() const 
{
  return (iDCC()-1) * kEBTowersPerSM + iTT() - 1;
}


std::ostream& operator<<(std::ostream& s,const EcalTrigTowerDetId& id) {
  return s << "(EcalTT subDet " << ((id.subDet()==EcalBarrel)?("Barrel"):("Endcap")) 
	   <<  " iz " << ((id.zside()>0)?("+ "):("- ")) << " ieta " 
	   << id.ietaAbs() << " iphi " << id.iphi() << ')';
}


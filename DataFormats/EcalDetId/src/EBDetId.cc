#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include <stdexcept>
//#include <iostream>

EBDetId::EBDetId() : DetId() {
}
  
EBDetId::EBDetId(uint32_t rawid) : DetId(rawid) {
}
  
EBDetId::EBDetId(int index1, int index2, int mode)  throw(std::runtime_error)
  : DetId(Ecal,EcalBarrel)
{
  int crystal_ieta;
  int crystal_iphi;
  if (mode == ETAPHIMODE) {
    crystal_ieta = index1;
    crystal_iphi = index2;  
  } else if (mode == SMCRYSTALMODE) {
    int SM = index1;
    int crystal = index2;
    int i = (int)  floor((crystal-1) / kCrystalsInPhi);
    int j = ((crystal-1) - (kCrystalsInPhi*i));
    if (SM <= 18) {
      crystal_ieta = i + 1;
      crystal_iphi = ((SM-1) * kCrystalsInPhi) + (kCrystalsInPhi-j);
    } else {
      crystal_ieta = -(i+1);
      crystal_iphi = ((SM-19) * kCrystalsInPhi) + j+1;
    }
  } else {
    throw(std::runtime_error("EBDetId:  Cannot create object.  Unknown mode for (int, int) constructor."));
  }

  if (crystal_ieta < -MAX_IETA || crystal_ieta == 0 || crystal_ieta > MAX_IETA ||
      crystal_iphi < MIN_IPHI || crystal_iphi > MAX_IPHI) {
    //    std::cout << "crystal_eta " << crystal_ieta << "crystal_phi " << crystal_iphi << std::endl;
    throw(std::runtime_error("EBDetId:  Cannot create object.  Indexes out of bounds."));
  }
  id_|=((crystal_ieta>0)?(0x10000|(crystal_ieta<<9)):((-crystal_ieta)<<9))|(crystal_iphi&0x1FF);
}
  
EBDetId::EBDetId(const DetId& gen) {
  if (!gen.null() && ( gen.det()!=Ecal || gen.subdetId()!=EcalBarrel )) {
    throw new std::exception();
  }
  id_=gen.rawId();
}
  
EBDetId& EBDetId::operator=(const DetId& gen) {
  if (!gen.null() && ( gen.det()!=Ecal || gen.subdetId()!=EcalBarrel )) {
    throw new std::exception();
  }
  id_=gen.rawId();
  return *this;
}

//New SM numbering scheme. Avoids discontinuity in phi crossing \eta=0  
int EBDetId::ism() const {
  int id = ( iphi() - 1 ) / kCrystalsInPhi + 1;
  if ( zside() < 0 ) id += 18;
  return id;
}

//Following TB 2004  numbering scheme 
int EBDetId::ic() const {
  int ie = ietaAbs() -1;
  int ip;
  if (zside() < 0) {
    ip = (( iphi() -1 ) % kCrystalsInPhi ) + 1;
  } else {
    ip = kCrystalsInPhi - ((iphi() -1 ) % kCrystalsInPhi );
  }

  return (ie * kCrystalsInPhi) + ip;
}

//Maintains SM crystals in bunch of 1700 indices
int EBDetId::hashedIndex() const {
  return (ism()-1) * kCrystalsPerSM + ic() -1;
}

//corrects for HB/EB differing iphi=1
int EBDetId::tower_iphi() const { 
  int iphi_simple=((iphi()-1)/5)+1; 
  iphi_simple-=2;
  return (iphi_simple<=0)?(iphi_simple+72):(iphi_simple);
}
  
std::ostream& operator<<(std::ostream& s,const EBDetId& id) {
  return s << "(EB ieta " << id.ieta() << ", iphi" << id.iphi() 
	   << " ; ism " << id.ism() << " , ic " << id.ic()  << ')';
}
  

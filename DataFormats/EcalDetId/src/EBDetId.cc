#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

//#include <iostream>
const int EBDetId::kModuleBoundaries[4] = { 25, 45, 65, 85 };

  
EBDetId::EBDetId(int index1, int index2, int mode) 
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
    throw cms::Exception("InvalidDetId") << "EBDetId:  Cannot create object.  Unknown mode for (int, int) constructor."; 
  }

  if ( !validDetId(crystal_ieta, crystal_iphi) ) {
    //    std::cout << "crystal_eta " << crystal_ieta << "crystal_phi " << crystal_iphi << std::endl;
    throw cms::Exception("InvalidDetId") << "EBDetId:  Cannot create object.  Indexes out of bounds \n" 
                                         << "eta = " << crystal_ieta << " phi = " << crystal_iphi;
  }
  id_|=((crystal_ieta>0)?(0x10000|(crystal_ieta<<9)):((-crystal_ieta)<<9))|(crystal_iphi&0x1FF);
}
  
EBDetId::EBDetId(const DetId& gen) {
  if (!gen.null() && ( gen.det()!=Ecal || gen.subdetId()!=EcalBarrel )) {
    throw cms::Exception("InvalidDetId");
  }
  id_=gen.rawId();
}

bool EBDetId::validDetId(int i, int j) {

  bool valid = true;
  if (i < -MAX_IETA || i == 0 || i > MAX_IETA ||
      j < MIN_IPHI || j > MAX_IPHI) {
    valid = false;
  }  
  return valid;

}
  
EBDetId& EBDetId::operator=(const DetId& gen) {
  if (!gen.null() && ( gen.det()!=Ecal || gen.subdetId()!=EcalBarrel )) {
    throw cms::Exception("InvalidDetId");
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

int EBDetId::im() const {
  for (int i=0; i < 4 ; i++)
    if ( ietaAbs() <= kModuleBoundaries[i] )
      return i+1;
  //Shold never be reached!
  return -1;
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
int EBDetId::numberBySM() const {
  return (ism()-1) * kCrystalsPerSM + ic() -1;
}

//corrects for HB/EB differing iphi=1
int EBDetId::tower_iphi() const { 
  int iphi_simple=((iphi()-1)/5)+1; 
  iphi_simple-=2;
  return ((iphi_simple<=0)?(iphi_simple+72):(iphi_simple));
}
  
std::ostream& operator<<(std::ostream& s,const EBDetId& id) {
  return s << "(EB ieta " << id.ieta() << ", iphi" << id.iphi() 
	   << " ; ism " << id.ism() << " , ic " << id.ic()  << ')';
}
  

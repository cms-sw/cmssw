#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

//#include <iostream>
#include <algorithm>
const int EBDetId::kModuleBoundaries[4] = { 25, 45, 65, 85 };

// pi / 180.
const float EBDetId::crystalUnitToEta = 0.017453292519943295;

 
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
  


//Following TB 2004  numbering scheme 
int EBDetId::ic() const {
  int ie = ietaAbs() -1;
  return  (ie * kCrystalsInPhi)
    +  ( positiveZ() ?
	 ( kCrystalsInPhi - ( (iphi() -1 ) % kCrystalsInPhi ) )
	 : ( ( iphi() -1 ) % kCrystalsInPhi  + 1)  
	 );
}


//Maintains SM crystals in bunch of 1700 indices
int EBDetId::numberBySM() const {
  return (ism()-1) * kCrystalsPerSM + ic() -1;
}

EBDetId EBDetId::offsetBy(int nrStepsEta, int nrStepsPhi ) const
{
        int newEta = ieta()+nrStepsEta;
        if( newEta*ieta() <= 0 ) {
                if( ieta() < 0 ) {
                        newEta++;
                } else if ( ieta() > 0 ) {
                        newEta--;
                }
        }
        int newPhi = iphi() + nrStepsPhi;
        while ( newPhi>360 ) newPhi -= 360;
        while ( newPhi<=0  ) newPhi += 360;

        if( validDetId( newEta, newPhi ) ) {
                return EBDetId( newEta, newPhi);
        } else {
                return EBDetId(0);
        }
}

EBDetId EBDetId::switchZSide() const
{
        int newEta = ieta()*-1;
        if( validDetId( newEta, iphi() ) ) {
                return EBDetId( newEta, iphi() );
        } else {
                return EBDetId(0);
        }
}


DetId EBDetId::offsetBy(const DetId startId, int nrStepsEta, int nrStepsPhi )
{
        if( startId.det() == DetId::Ecal && startId.subdetId() == EcalBarrel ) {
                EBDetId ebStartId(startId);
                return ebStartId.offsetBy( nrStepsEta, nrStepsPhi ).rawId();
        } else {
                return DetId(0);
        }
}

DetId EBDetId::switchZSide( const DetId startId )
{
        if( startId.det() == DetId::Ecal && startId.subdetId() == EcalBarrel ) {
                EBDetId ebStartId(startId);
                return ebStartId.switchZSide().rawId();
        } else {
                return DetId(0);
        }
}

//corrects for HB/EB differing iphi=1
int EBDetId::tower_iphi() const { 
  int iphi_simple=((iphi()-1)/5)+1; 
  iphi_simple-=2;
  return ((iphi_simple<=0)?(iphi_simple+72):(iphi_simple));
}


bool EBDetId::isNextToBoundary(EBDetId id) {
  return isNextToEtaBoundary( id ) || isNextToPhiBoundary( id );
}

bool EBDetId::isNextToEtaBoundary(EBDetId id) {
  int ieta = id.ietaSM();
  return ieta == 1 || std::find( kModuleBoundaries, kModuleBoundaries + 4, ieta );
}

bool EBDetId::isNextToPhiBoundary(EBDetId id) {
  int iphi = id.iphiSM();
  return iphi == 1 || iphi == 20;
}

int EBDetId::distanceEta(const EBDetId& a,const EBDetId& b)
{
  if (a.ieta() * b.ieta() > 0)
    return abs(a.ieta()-b.ieta());
  else
    return abs(a.ieta()-b.ieta())-1;
}

int EBDetId::distancePhi(const EBDetId& a,const EBDetId& b) {
  int PI = 180;
  int  result = a.iphi() - b.iphi();
  
  while  (result > PI)    result -= 2*PI;
  while  (result <= -PI)  result += 2*PI;
  return abs(result);

}

float EBDetId::approxEta( const DetId id ) {
  if( id.subdetId() == EcalBarrel ) {
    EBDetId ebId( id );
    return ebId.approxEta();
  } else {
    return 0;
  }
}
  
std::ostream& operator<<(std::ostream& s,const EBDetId& id) {
  return s << "(EB ieta " << id.ieta() << ", iphi " << id.iphi() 
	   << " ; ism " << id.ism() << " , ic " << id.ic()  << ')';
}
  

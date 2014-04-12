//
// HcalHPDRBXMap.cc
//
//   description: implementation of HcalHPDRBXMap.
//
//   author: J.P. Chou, Brown
//

#include "RecoMET/METAlgorithms/interface/HcalHPDRBXMap.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

// empty constructor/destructor
HcalHPDRBXMap::HcalHPDRBXMap() {}
HcalHPDRBXMap::~HcalHPDRBXMap() {}

// returns if the HPD index is valid
bool HcalHPDRBXMap::isValidHPD(int index)
{
  return (index>=0 && index<=NUM_HPDS-1);
}

// returns if the RBX index is valid
bool HcalHPDRBXMap::isValidRBX(int index)
{
  return (index>=0 && index<=NUM_RBXS-1);
}

bool HcalHPDRBXMap::isValid(const HcalDetId& id)
{
  if(id.subdet()!=HcalBarrel && id.subdet()!=HcalEndcap) return false;
  return isValid(id.ieta(),id.iphi());
}

bool HcalHPDRBXMap::isValid(int ieta, int iphi)
{
  int absieta=abs(ieta);
  if(absieta<=29 && absieta>=1 && iphi>=1 && iphi<=72) {
    if(absieta<=20) return true;
    if(absieta>=21 && iphi%2==1) return true;
  }
  return false;
}

// returns the subdetector (HE or HE) for an HPD index
HcalSubdetector HcalHPDRBXMap::subdetHPD(int index)
{
  if(!isValidHPD(index))
    throw edm::Exception(edm::errors::LogicError)
      << " HPD index " << index << " is invalid in HcalHPDRBXMap::subdetHPD().\n";

  if(index/NUM_HPDS_PER_SUBDET<=1) return HcalBarrel;
  else return HcalEndcap;
}

// returns the subdetector (HE or HE) for an RBX index
HcalSubdetector HcalHPDRBXMap::subdetRBX(int index)
{
  if(!isValidRBX(index))
    throw edm::Exception(edm::errors::LogicError)
      << " RBX index " << index << " is invalid in HcalHPDRBXMap::subdetRBX().\n";

  if(index/NUM_RBXS_PER_SUBDET<=1) return HcalBarrel;
  else return HcalEndcap;
}

// returns the zside (1 or -1) given an HPD index
int HcalHPDRBXMap::zsideHPD(int index)
{
  if(!isValidHPD(index))
    throw edm::Exception(edm::errors::LogicError)
      << " HPD index " << index << " is invalid in HcalHPDRBXMap::zsideHPD().\n";

  if(index/NUM_HPDS_PER_SUBDET==0 || index/NUM_HPDS_PER_SUBDET==2) return 1;
  else return -1;
}

// returns the zside (1 or -1) given an RBX index
int HcalHPDRBXMap::zsideRBX(int index)
{
  if(!isValidRBX(index))
    throw edm::Exception(edm::errors::LogicError)
      << " RBX index " << index << " is invalid in HcalHPDRBXMap::zsideRBX().\n";

  if(index/NUM_RBXS_PER_SUBDET==0 || index/NUM_RBXS_PER_SUBDET==2) return 1;
  else return -1;
}

// returns the lowest iphi used in an HPD
int HcalHPDRBXMap::iphiloHPD(int index)
{
  if(!isValidHPD(index))
    throw edm::Exception(edm::errors::LogicError)
      << " HPD index " << index << " is invalid in HcalHPDRBXMap::iphiloHPD().\n";
  
  // adjust for offset between iphi and the HPD index
  // index-->iphi
  // 0-->71, 1-->72, 2-->1, 3-->2, 4-->3, ..., 70-->69, 71-->70
  int iphi=index%NUM_HPDS_PER_SUBDET-1;
  if(iphi<=0) iphi+=NUM_HPDS_PER_SUBDET;

  // HB
  if(subdetHPD(index)==HcalBarrel) return iphi;

  // HE
  if(iphi%2==0) return iphi-1;
  else          return iphi;
}

// returns the lowest iphi used in an RBX
int HcalHPDRBXMap::iphiloRBX(int index)
{
  if(!isValidRBX(index))
    throw edm::Exception(edm::errors::LogicError)
      << " RBX index " << index << " is invalid in HcalHPDRBXMap::iphiloRBX().\n";

  // get the list of HPD indices in the RBX
  boost::array<int, NUM_HPDS_PER_RBX> arr;
  indicesHPDfromRBX(index, arr);

  // return the lowest iphi of the first HPD
  return iphiloHPD(arr[0]);
}

// returns the highest iphi used in an HPD
int HcalHPDRBXMap::iphihiHPD(int index)
{
  if(!isValidHPD(index))
    throw edm::Exception(edm::errors::LogicError)
      << " HPD index " << index << " is invalid in HcalHPDRBXMap::iphihiHPD().\n";
  
  // adjust for offset between iphi and the HPD index
  // index-->iphi
  // 0-->71, 1-->72, 2-->1, 3-->2, 4-->3, ..., 70-->69, 71-->70
  int iphi=index%NUM_HPDS_PER_SUBDET-1;
  if(iphi<=0) iphi+=NUM_HPDS_PER_SUBDET;

  // HB
  if(subdetHPD(index)==HcalBarrel) return iphi;

  // HE
  if(iphi%2==0) return iphi;
  else          return iphi+1;
}

// returns the highest iphi used in an RBX
int HcalHPDRBXMap::iphihiRBX(int index)
{
  if(!isValidRBX(index))
    throw edm::Exception(edm::errors::LogicError)
      << " RBX index " << index << " is invalid in HcalHPDRBXMap::iphihiRBX().\n";
  
  // get the list of HPD indices in the RBX
  boost::array<int, NUM_HPDS_PER_RBX> arr;
  indicesHPDfromRBX(index, arr);

  // return the highest iphi of the last HPD
  return iphihiHPD(arr[NUM_HPDS_PER_RBX-1]);
}


// returns the list of HPD indices found in a given RBX
void HcalHPDRBXMap::indicesHPDfromRBX(int rbxindex, boost::array<int, NUM_HPDS_PER_RBX>& hpdindices)
{
  if(!isValidRBX(rbxindex))
    throw edm::Exception(edm::errors::LogicError)
      << " RBX index " << rbxindex << " is invalid in HcalHPDRBXMap::indicesHPD().\n";

  for(unsigned int i=0; i<hpdindices.size(); i++)
    hpdindices[i]=rbxindex*NUM_HPDS_PER_RBX+i;

  return;
}

// returns the RBX index given an HPD index
int HcalHPDRBXMap::indexRBXfromHPD(int hpdindex)
{
  if(!isValidHPD(hpdindex))
    throw edm::Exception(edm::errors::LogicError)
      << " HPD index " << hpdindex << " is invalid in HcalHPDRBXMap::indexRBX().\n";

  return hpdindex/NUM_HPDS_PER_RBX;
}


// get the HPD index from an HcalDetector id
int HcalHPDRBXMap::indexHPD(const HcalDetId& id)
{
  // return bad index if subdetector is invalid
  if(!isValid(id)) {
    throw edm::Exception(edm::errors::LogicError)
      << " HcalDetId " << id << " is invalid in HcalHPDRBXMap::indexHPD().\n";
  }

  // specify the readout module (subdet and number)
  int subdet=-1;
  if(id.subdet()==HcalBarrel && id.zside()==1)  subdet=0;
  if(id.subdet()==HcalBarrel && id.zside()==-1) subdet=1;
  if(id.subdet()==HcalEndcap && id.zside()==1)  subdet=2;
  if(id.subdet()==HcalEndcap && id.zside()==-1) subdet=3;

  int iphi=id.iphi();
  int absieta=abs(id.ieta());

  // adjust for offset between iphi and the HPD index
  // index-->iphi
  // 0-->71, 1-->72, 2-->1, 3-->2, 4-->3, ..., 70-->69, 71-->70
  int index=iphi+1;
  if(index>=NUM_HPDS_PER_SUBDET) index-=NUM_HPDS_PER_SUBDET;
  index+=subdet*NUM_HPDS_PER_SUBDET;

  // modify the index in the HE
  if((subdet==2 || subdet==3) && absieta>=21 && absieta<=29) {
    if(iphi%4==3 && absieta%2==1 && absieta!=29) index++;
    if(iphi%4==3 && absieta==29 && id.depth()==2) index++;
    if(iphi%4==1 && absieta%2==0 && absieta!=29) index++;
    if(iphi%4==1 && absieta==29 && id.depth()==1) index++;
  }
  return index;
}

int HcalHPDRBXMap::indexRBX(const HcalDetId& id)
{
  return indexRBXfromHPD(indexHPD(id));
}

void HcalHPDRBXMap::indexHPDfromEtaPhi(int ieta, int iphi, std::vector<int>& hpdindices)
{
  // clear the vector
  hpdindices.clear();
  int absieta=abs(ieta);

  if(absieta<=15) {        // HB only, depth doesn't matter
    hpdindices.push_back(indexHPD(HcalDetId(HcalBarrel, ieta, iphi, 1)));
  } else if(absieta==16) { // HB and HE, depth doesn't matter
    hpdindices.push_back(indexHPD(HcalDetId(HcalBarrel, ieta, iphi, 1)));
    hpdindices.push_back(indexHPD(HcalDetId(HcalEndcap, ieta, iphi, 3)));
  } else if(absieta<29) {  // HE only, depth doesn't matter
    hpdindices.push_back(indexHPD(HcalDetId(HcalEndcap, ieta, iphi, 1)));
  } else {                 // HE only, but depth matters
    hpdindices.push_back(indexHPD(HcalDetId(HcalEndcap, ieta, iphi, 1)));
    hpdindices.push_back(indexHPD(HcalDetId(HcalEndcap, ieta, iphi, 2)));
  }

  return;
}

void HcalHPDRBXMap::indexRBXfromEtaPhi(int ieta, int iphi, std::vector<int>& rbxindices)
{
  // clear the vector
  rbxindices.clear();
  int absieta=abs(ieta);

  if(absieta<=15) {        // HB only
    rbxindices.push_back(indexRBX(HcalDetId(HcalBarrel, ieta, iphi, 1)));
  } else if(absieta==16) { // HB and HE
    rbxindices.push_back(indexRBX(HcalDetId(HcalBarrel, ieta, iphi, 1)));
    rbxindices.push_back(indexRBX(HcalDetId(HcalEndcap, ieta, iphi, 3)));
  } else {                 // HE only
    rbxindices.push_back(indexRBX(HcalDetId(HcalEndcap, ieta, iphi, 1)));
  }

  return;
}

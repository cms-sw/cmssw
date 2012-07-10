//
// HcalNoiseRBXArray.cc
//
//   description: implementation of the HcalNoiseRBXArray
//
//   author: J.P. Chou, Brown
//
//

#include "RecoMET/METAlgorithms/interface/HcalNoiseRBXArray.h"

using namespace reco;

// constructor sets the idnumbers for the rbx's and the hpd's
HcalNoiseRBXArray::HcalNoiseRBXArray()
{
  for(unsigned int i=0; i<size(); i++) {
    HcalNoiseRBX& rbx=at(i);
    
    // set the rbxnumber here
    rbx.idnumber_=i;

    // set the hpdnumber here
    boost::array<int, HcalHPDRBXMap::NUM_HPDS_PER_RBX> hpdindices;
    HcalHPDRBXMap::indicesHPDfromRBX(i, hpdindices);
    for(int j=0; j<HcalHPDRBXMap::NUM_HPDS_PER_RBX; j++) {
      rbx.hpds_[j].idnumber_=hpdindices[j];
    }
  }
}

HcalNoiseRBXArray::~HcalNoiseRBXArray()
{
}

std::vector<HcalNoiseHPD>::const_iterator HcalNoiseRBXArray::endHPD(void) const
{
  // the choice of which rbx to use is arbitrary,
  // as long as we're consistent
  return at(0).hpds_.end();
}

// code here should be same as above (modulo 'const'ness)
std::vector<HcalNoiseHPD>::iterator HcalNoiseRBXArray::endHPD(void)
{
  // the choice of which rbx to use is arbitrary,
  // as long as we're consistent
  return at(0).hpds_.end();
}

std::vector<HcalNoiseHPD>::iterator HcalNoiseRBXArray::findHPD(int hpdindex)
{
  // if the hpdindex is invalid
  if(!HcalHPDRBXMap::isValidHPD(hpdindex)) return endHPD();
  
  int rbxindex=HcalHPDRBXMap::indexRBXfromHPD(hpdindex);
  
  // find the HPD in the RBX
  HcalNoiseRBX& rbx=at(rbxindex);
  for(std::vector<HcalNoiseHPD>::iterator it=rbx.hpds_.begin(); it!=rbx.hpds_.end(); ++it) {
    if(it->idnumber_==hpdindex) return it;
  }
  
  // if we're here, this is a bug
  throw edm::Exception(edm::errors::LogicError)
    << "Could not find hpdindex " << hpdindex << " in HcalNoiseRBXArray::findHPDfromDetID().  This is a bug.\n";
  return endHPD();
}

// code here should be same as above (modulo 'const'ness)
std::vector<HcalNoiseHPD>::const_iterator HcalNoiseRBXArray::findHPD(int hpdindex) const
{
  // if the hpdindex is invalid
  if(!HcalHPDRBXMap::isValidHPD(hpdindex)) return endHPD();
  
  int rbxindex=HcalHPDRBXMap::indexRBXfromHPD(hpdindex);
  
  // find the HPD in the RBX
  const HcalNoiseRBX& rbx=at(rbxindex);
  for(std::vector<HcalNoiseHPD>::const_iterator it=rbx.hpds_.begin(); it!=rbx.hpds_.end(); ++it) {
    if(it->idnumber_==hpdindex) return it;
  }
  
  // if we're here, this is a bug
  throw edm::Exception(edm::errors::LogicError)
    << "Could not find hpdindex " << hpdindex << " in HcalNoiseRBXArray::findHPDfromDetID().  This is a bug.\n";
  return endHPD();
}

HcalNoiseRBXArray::iterator
HcalNoiseRBXArray::findRBX(int rbxindex)
{
  if(!HcalHPDRBXMap::isValidRBX(rbxindex)) return endRBX();
  return begin()+rbxindex;
}

HcalNoiseRBXArray::const_iterator
HcalNoiseRBXArray::findRBX(int rbxindex) const
{
  if(!HcalHPDRBXMap::isValidRBX(rbxindex)) return endRBX();
  return begin()+rbxindex;
}

std::vector<HcalNoiseHPD>::iterator
HcalNoiseRBXArray::findHPD(const HcalDetId& id)
{
  if(!HcalHPDRBXMap::isValid(id)) return endHPD();
  return findHPD(HcalHPDRBXMap::indexHPD(id));
}

std::vector<HcalNoiseHPD>::const_iterator
HcalNoiseRBXArray::findHPD(const HcalDetId& id) const
{
  if(!HcalHPDRBXMap::isValid(id)) return endHPD();
  return findHPD(HcalHPDRBXMap::indexHPD(id));
}

HcalNoiseRBXArray::iterator
HcalNoiseRBXArray::findRBX(const HcalDetId& id)
{
  if(!HcalHPDRBXMap::isValid(id)) return endRBX();
  return findRBX(HcalHPDRBXMap::indexRBX(id));
}

HcalNoiseRBXArray::const_iterator
HcalNoiseRBXArray::findRBX(const HcalDetId& id) const
{
  if(!HcalHPDRBXMap::isValid(id)) return endRBX();
  return findRBX(HcalHPDRBXMap::indexRBX(id));
}

std::vector<HcalNoiseHPD>::iterator
HcalNoiseRBXArray::findHPD(const HBHEDataFrame& f)
{ return findHPD(f.id()); }

std::vector<HcalNoiseHPD>::const_iterator
HcalNoiseRBXArray::findHPD(const HBHEDataFrame& f) const
{ return findHPD(f.id()); }

HcalNoiseRBXArray::iterator
HcalNoiseRBXArray::findRBX(const HBHEDataFrame& f)
{ return findRBX(f.id()); }

HcalNoiseRBXArray::const_iterator
HcalNoiseRBXArray::findRBX(const HBHEDataFrame& f) const
{ return findRBX(f.id()); }

std::vector<HcalNoiseHPD>::iterator
HcalNoiseRBXArray::findHPD(const HBHERecHit& h)
{ return findHPD(h.id()); }

std::vector<HcalNoiseHPD>::const_iterator
HcalNoiseRBXArray::findHPD(const HBHERecHit& h) const
{ return findHPD(h.id()); }

HcalNoiseRBXArray::iterator
HcalNoiseRBXArray::findRBX(const HBHERecHit& h)
{ return findRBX(h.id()); }

HcalNoiseRBXArray::const_iterator
HcalNoiseRBXArray::findRBX(const HBHERecHit& h) const
{ return findRBX(h.id()); }


void HcalNoiseRBXArray::findHPD(const CaloTower& tower, std::vector<std::vector<HcalNoiseHPD>::const_iterator>& vec) const
{
  // clear the vector
  vec.clear();

  // check if the tower corresponds to a valid HPD/RBX
  if(!HcalHPDRBXMap::isValid(tower.ieta(), tower.iphi())) return;

  // find the HPD indices
  std::vector<int> hpdindices;
  HcalHPDRBXMap::indexHPDfromEtaPhi(tower.ieta(), tower.iphi(), hpdindices);
  for(std::vector<int>::const_iterator it=hpdindices.begin(); it!=hpdindices.end(); ++it)
    vec.push_back(findHPD(*it));

  return;
}

void HcalNoiseRBXArray::findHPD(const CaloTower& tower, std::vector<std::vector<HcalNoiseHPD>::iterator>& vec)
{
  // clear the vector
  vec.clear();

  // check if the tower corresponds to a valid HPD/RBX
  if(!HcalHPDRBXMap::isValid(tower.ieta(), tower.iphi())) return;

  // find the HPD indices
  std::vector<int> hpdindices;
  HcalHPDRBXMap::indexHPDfromEtaPhi(tower.ieta(), tower.iphi(), hpdindices);
  for(std::vector<int>::const_iterator it=hpdindices.begin(); it!=hpdindices.end(); ++it)
    vec.push_back(findHPD(*it));

  return;
}

void HcalNoiseRBXArray::findRBX(const CaloTower& tower, std::vector<HcalNoiseRBXArray::iterator>& vec)
{
  // clear the vector
  vec.clear();

  // check if the tower corresponds to a valid HPD/RBX
  if(!HcalHPDRBXMap::isValid(tower.ieta(), tower.iphi())) return;

  // find the RBX indices
  std::vector<int> rbxindices;
  HcalHPDRBXMap::indexRBXfromEtaPhi(tower.ieta(), tower.iphi(), rbxindices);
  for(std::vector<int>::const_iterator it=rbxindices.begin(); it!=rbxindices.end(); ++it)
    vec.push_back(findRBX(*it));

  return;
}

void HcalNoiseRBXArray::findRBX(const CaloTower& tower, std::vector<HcalNoiseRBXArray::const_iterator>& vec) const
{
  // clear the vector
  vec.clear();

  // check if the tower corresponds to a valid HPD/RBX
  if(!HcalHPDRBXMap::isValid(tower.ieta(), tower.iphi())) return;

  // find the RBX indices
  std::vector<int> rbxindices;
  HcalHPDRBXMap::indexRBXfromEtaPhi(tower.ieta(), tower.iphi(), rbxindices);
  for(std::vector<int>::const_iterator it=rbxindices.begin(); it!=rbxindices.end(); ++it)
    vec.push_back(findRBX(*it));

  return;
}

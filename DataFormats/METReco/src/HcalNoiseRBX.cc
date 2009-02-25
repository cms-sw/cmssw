//
// HcalNoiseRBX.cc
//
//   description: container class of RBX information for studying the HCAL Noise
//
//   author: J.P. Chou, Brown
//
//

#include "DataFormats/METReco/interface/HcalNoiseRBX.h"


using namespace reco;

// default constructor
HcalNoiseRBX::HcalNoiseRBX() :
  idnumber_(0), twrHadE_(0.0), twrEmE_(0.0)
{
}

// destructor
HcalNoiseRBX::~HcalNoiseRBX()
{
}

// accessors
int HcalNoiseRBX::idnumber(void) const
{
  return idnumber_;
}

HcalSubdetector HcalNoiseRBX::subdet(void) const
{
  return HcalHPDRBXMap::subdetRBX(idnumber_);
}

int HcalNoiseRBX::zside(void) const
{
  return HcalHPDRBXMap::zsideRBX(idnumber_);
}

int HcalNoiseRBX::iphilo(void) const
{
  return HcalHPDRBXMap::iphiloRBX(idnumber_);
}

int HcalNoiseRBX::iphihi(void) const
{
  return HcalHPDRBXMap::iphihiRBX(idnumber_);
}

int HcalNoiseRBX::numHits(void) const
{
  int tot=0;
  for(unsigned int i=0; i<hpds_.size(); i++)
    tot += hpds_[i].numHits();
  return tot;
}

int HcalNoiseRBX::numHitsAboveThreshold(void) const
{
  int tot=0;
  for(unsigned int i=0; i<hpds_.size(); i++)
    tot += hpds_[i].numHitsAboveThreshold();
  return tot;
}

int HcalNoiseRBX::totalZeros(void) const
{
  int tot=0;
  for(unsigned int i=0; i<hpds_.size(); i++)
    tot += hpds_[i].totalZeros();
  return tot;
}

int HcalNoiseRBX::maxZeros(void) const
{
  int max=0;
  for(unsigned int i=0; i<hpds_.size(); i++)
    if(hpds_[i].maxZeros()>max)
      max=hpds_[i].maxZeros();
  return max;
}

HcalNoiseHPDArray::const_iterator HcalNoiseRBX::maxHPD(void) const
{
  HcalNoiseHPDArray::const_iterator maxit=hpds_.begin();
  for(HcalNoiseHPDArray::const_iterator it=hpds_.begin(); it!=hpds_.end(); ++it)
    if(it->rechitEnergy()>maxit->rechitEnergy())
      maxit=it;
  return maxit;
}

HcalNoiseHPDArray::const_iterator HcalNoiseRBX::beginHPD(void) const
{
  return hpds_.begin();
}

HcalNoiseHPDArray::const_iterator HcalNoiseRBX::endHPD(void) const
{
  return hpds_.end();
}

HcalNoiseHPDArray HcalNoiseRBX::HPDs(void) const
{
  return hpds_;
}

double HcalNoiseRBX::rechitEnergy(void) const
{
  double total=0;
  for(unsigned int i=0; i<hpds_.size(); i++)
    total += hpds_[i].rechitEnergy();
  return total;
}

double HcalNoiseRBX::caloTowerHadE(void) const
{
  return twrHadE_;
}

double HcalNoiseRBX::caloTowerEmE(void) const
{
  return twrEmE_;
}

double HcalNoiseRBX::caloTowerTotalE(void) const
{
  return twrEmE_+twrHadE_;
}

double HcalNoiseRBX::caloTowerEmFraction(void) const
{
  return caloTowerTotalE()!=0.0 ? twrEmE_/caloTowerTotalE() : -999.;
}

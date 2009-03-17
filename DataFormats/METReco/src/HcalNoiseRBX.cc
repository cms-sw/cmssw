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
  idnumber_(0), hpds_(HcalHPDRBXMap::NUM_HPDS_PER_RBX)
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

const std::vector<HcalNoiseHPD>& HcalNoiseRBX::HPDs(void) const
{
  return hpds_;
}

std::vector<HcalNoiseHPD>::const_iterator HcalNoiseRBX::maxHPD(double threshold) const
{
  std::vector<HcalNoiseHPD>::const_iterator maxit=hpds_.end();
  double maxenergy=-99999999.;
  for(std::vector<HcalNoiseHPD>::const_iterator it=hpds_.begin(); it!=hpds_.end(); ++it) {
    double tempenergy=it->recHitEnergy();
    if(tempenergy>maxenergy) {
      maxenergy=tempenergy;
      maxit=it;
    }
  }
  return maxit;
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

double HcalNoiseRBX::recHitEnergy(double threshold) const
{
  double total=0;
  for(unsigned int i=0; i<hpds_.size(); i++)
    total += hpds_[i].recHitEnergy(threshold);
  return total;
}

double HcalNoiseRBX::minRecHitTime(double threshold) const
{
  double mintime=9999999.;
  for(unsigned int i=0; i<hpds_.size(); i++) {
    double temptime=hpds_[i].minRecHitTime(threshold);
    if(temptime<mintime) mintime=temptime;
  }
  return mintime;
}

double HcalNoiseRBX::maxRecHitTime(double threshold) const
{
  double maxtime=-9999999.;
  for(unsigned int i=0; i<hpds_.size(); i++) {
    double temptime=hpds_[i].maxRecHitTime(threshold);
    if(temptime>maxtime) maxtime=temptime;
  }
  return maxtime;
}

int HcalNoiseRBX::numRecHits(double threshold) const
{
  int total=0;
  for(unsigned int i=0; i<hpds_.size(); i++)
    total += hpds_[i].numRecHits(threshold);
  return total;
}

/*double HcalNoiseRBX::caloTowerHadE(void) const
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
*/

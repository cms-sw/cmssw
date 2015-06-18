//
// HcalNoiseHPD.cc
//
//   description: container class of HPD information for analyzing HCAL Noise
//
//   author: J.P. Chou, Brown
//
//

#include "DataFormats/METReco/interface/HcalNoiseHPD.h"


using namespace reco;

// default constructor
HcalNoiseHPD::HcalNoiseHPD()
  : idnumber_(0), totalZeros_(0), maxZeros_(0),
    bigCharge_(HBHEDataFrame::MAXSAMPLES, 0.0),
    big5Charge_(HBHEDataFrame::MAXSAMPLES, 0.0)
{
  // reserve some space, so that there's no reallocation issues
  rechits_.reserve(19);
  calotowers_.reserve(19);
}
  
// destructor
HcalNoiseHPD::~HcalNoiseHPD()
{
}
  
// accessors
int HcalNoiseHPD::idnumber(void) const
{
  return idnumber_;
}
  
const std::vector<float> HcalNoiseHPD::bigCharge(void) const
{
  return bigCharge_;
}
  
float HcalNoiseHPD::bigChargeTotal(void) const
{
  float total=0;
  for(unsigned int i=0; i<bigCharge_.size(); i++) {
    total += bigCharge_[i];
  }
  return total;
}

float HcalNoiseHPD::bigChargeHighest2TS(unsigned int firstts) const
{
  float total=0;
  for(unsigned int i=firstts; i<firstts+2 && i<bigCharge_.size(); i++)
    total += bigCharge_[i];
  return total;
}


float HcalNoiseHPD::bigChargeHighest3TS(unsigned int firstts) const
{
  float total=0;
  for(unsigned int i=firstts; i<firstts+3 && i<bigCharge_.size(); i++)
    total += bigCharge_[i];
  return total;
}
  
const std::vector<float> HcalNoiseHPD::big5Charge(void) const
{
  return big5Charge_;
}
  
float HcalNoiseHPD::big5ChargeTotal(void) const
{
  float total=0;
  for(unsigned int i=0; i<big5Charge_.size(); i++) {
    total += big5Charge_[i];
  }
  return total;
}
  
float HcalNoiseHPD::big5ChargeHighest2TS(unsigned int firstts) const
{
  float total=0;
  for(unsigned int i=firstts; i<firstts+2 && i<big5Charge_.size(); i++)
    total += big5Charge_[i];
  return total;
}
  
float HcalNoiseHPD::big5ChargeHighest3TS(unsigned int firstts) const
{
  float total=0;
  for(unsigned int i=firstts; i<firstts+2 && i<big5Charge_.size(); i++)
    total += big5Charge_[i];
  return total;
}
  
int HcalNoiseHPD::totalZeros(void) const
{
  return totalZeros_;
}
  
int HcalNoiseHPD::maxZeros(void) const
{
  return maxZeros_;
}

const edm::RefVector<HBHERecHitCollection> HcalNoiseHPD::recHits(void) const
{
  return rechits_;
}
  
float HcalNoiseHPD::recHitEnergy(const float threshold) const
{
  double total=0.0;
  for(edm::RefVector<HBHERecHitCollection>::const_iterator it=rechits_.begin(); it!=rechits_.end(); ++it) {
    const float energy=(*it)->eraw();
    if(energy>=threshold) total+=energy;
  }
  return total;
}

float HcalNoiseHPD::recHitEnergyFailR45(const float threshold) const
{
  double total=0.0;
  for(edm::RefVector<HBHERecHitCollection>::const_iterator it=rechits_.begin(); it!=rechits_.end(); ++it) {
    const float energy=(*it)->eraw();
    if((*it)->flagField(HcalCaloFlagLabels::HBHETS4TS5Noise))
       if(energy>=threshold) total+=energy;
  }
  return total;
}

float HcalNoiseHPD::minRecHitTime(const float threshold) const
{
  float mintime=9999999;
  for(edm::RefVector<HBHERecHitCollection>::const_iterator it=rechits_.begin(); it!=rechits_.end(); ++it) {
    if((*it)->energy()<threshold) continue;
    float time=(*it)->time();
    if(mintime>time) mintime=time;
  }
  return mintime;
}
  
float HcalNoiseHPD::maxRecHitTime(const float threshold) const
{
  float maxtime=-9999999;
  for(edm::RefVector<HBHERecHitCollection>::const_iterator it=rechits_.begin(); it!=rechits_.end(); ++it) {
    if((*it)->energy()<threshold) continue;
    float time=(*it)->time();
    if(maxtime<time) maxtime=time;
  }
  return maxtime;
}

int HcalNoiseHPD::numRecHits(const float threshold) const
{
  int count=0;
  for(edm::RefVector<HBHERecHitCollection>::const_iterator it=rechits_.begin(); it!=rechits_.end(); ++it)
    if((*it)->eraw()>=threshold) ++count;
  return count;
}

int HcalNoiseHPD::numRecHitsFailR45(const float threshold) const
{
  int count=0;
  for(edm::RefVector<HBHERecHitCollection>::const_iterator it=rechits_.begin(); it!=rechits_.end(); ++it)
    if((*it)->flagField(HcalCaloFlagLabels::HBHETS4TS5Noise))
      if((*it)->eraw()>=threshold) ++count;
  return count;
}

const edm::RefVector<CaloTowerCollection> HcalNoiseHPD::caloTowers(void) const
{
  return calotowers_;
}
  
double HcalNoiseHPD::caloTowerHadE(void) const
{
  double total=0;
  for(edm::RefVector<CaloTowerCollection>::const_iterator it=calotowers_.begin(); it!=calotowers_.end(); ++it)
    total += (*it)->hadEnergy();
  return total;
}
  
double HcalNoiseHPD::caloTowerEmE(void) const
{
  double total=0;
  for(edm::RefVector<CaloTowerCollection>::const_iterator it=calotowers_.begin(); it!=calotowers_.end(); ++it)
    total += (*it)->emEnergy();
  return total;
}

double HcalNoiseHPD::caloTowerTotalE(void) const
{
  double total=0;
  for(edm::RefVector<CaloTowerCollection>::const_iterator it=calotowers_.begin(); it!=calotowers_.end(); ++it)
    total += (*it)->emEnergy()+(*it)->hadEnergy();
  return total;
}
  
double HcalNoiseHPD::caloTowerEmFraction(void) const
{
  double h=0, e=0;
  for(edm::RefVector<CaloTowerCollection>::const_iterator it=calotowers_.begin(); it!=calotowers_.end(); ++it) {
    e += (*it)->emEnergy();
    h += (*it)->hadEnergy();
  }
  return (e+h)!=0 ? e/(e+h) : 999.;
}

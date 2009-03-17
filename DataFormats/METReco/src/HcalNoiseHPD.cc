//
// HcalNoiseHPD.cc
//
//   description: container class of HPD information for analyzing HCAL Noise
//
//   author: J.P. Chou, Brown
//
//

#include "DataFormats/METReco/interface/HcalHPDRBXMap.h"
#include "DataFormats/METReco/interface/HcalNoiseHPD.h"

#include "TMath.h"

using namespace reco;

// default constructor
HcalNoiseHPD::HcalNoiseHPD()
  : idnumber_(0), totalZeros_(0), maxZeros_(0),
    bigDigi_(HBHEDataFrame::MAXSAMPLES, 0.0),
    big5Digi_(HBHEDataFrame::MAXSAMPLES, 0.0),
    allDigi_(HBHEDataFrame::MAXSAMPLES, 0.0)
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
  
HcalSubdetector HcalNoiseHPD::subdet(void) const
{
  return HcalHPDRBXMap::subdetHPD(idnumber_);
}
  
int HcalNoiseHPD::zside(void) const
{
  return HcalHPDRBXMap::zsideHPD(idnumber_);
}
  
int HcalNoiseHPD::iphilo(void) const
{
  return HcalHPDRBXMap::iphiloHPD(idnumber_);
}
  
int HcalNoiseHPD::iphihi(void) const
{
  return HcalHPDRBXMap::iphihiHPD(idnumber_);
}
  
const std::vector<float>& HcalNoiseHPD::bigDigi(void) const
{
  return bigDigi_;
}
  
float HcalNoiseHPD::bigDigiTime(void) const
{
  float total=0, time=0;
  for(unsigned int i=0; i<bigDigi_.size(); i++) {
    time  += i*bigDigi_[i];
    total += bigDigi_[i];
  }
  return total!=0.0 ? time/total : -999.;
}
  
float HcalNoiseHPD::bigDigiTotal(void) const
{
  float total=0;
  for(unsigned int i=0; i<bigDigi_.size(); i++) {
    total += bigDigi_[i];
  }
  return total;
}
  
float HcalNoiseHPD::bigDigiHighest2TS(void) const
{
  float total=0;
  for(unsigned int i=0; i<bigDigi_.size()-1; i++) {
    float temp = bigDigi_[i]+bigDigi_[i+1];
    if(temp>total) total=temp;
  }
  return total;
}
  
float HcalNoiseHPD::bigDigiHighest3TS(void) const
{
  float total=0;
  for(unsigned int i=0; i<bigDigi_.size()-2; i++) {
    float temp = bigDigi_[i]+bigDigi_[i+1]+bigDigi_[i+2];
    if(temp>total) total=temp;
  }
  return total;
}
  
const std::vector<float>& HcalNoiseHPD::big5Digi(void) const
{
  return big5Digi_;
}
  
float HcalNoiseHPD::big5DigiTime(void) const
{
  float total=0, time=0;
  for(unsigned int i=0; i<big5Digi_.size(); i++) {
    time  += i*big5Digi_[i];
    total += big5Digi_[i];
  }
  return total!=0.0 ? time/total : -999.;
}
  
float HcalNoiseHPD::big5DigiTotal(void) const
{
  float total=0;
  for(unsigned int i=0; i<big5Digi_.size(); i++) {
    total += big5Digi_[i];
  }
  return total;
}
  
float HcalNoiseHPD::big5DigiHighest2TS(void) const
{
  float total=0;
  for(unsigned int i=0; i<big5Digi_.size()-1; i++) {
    float temp = big5Digi_[i]+big5Digi_[i+1];
    if(temp>total) total=temp;
  }
  return total;
}
  
float HcalNoiseHPD::big5DigiHighest3TS(void) const
{
  float total=0;
  for(unsigned int i=0; i<big5Digi_.size()-2; i++) {
    float temp = big5Digi_[i]+big5Digi_[i+1]+big5Digi_[i+2];
    if(temp>total) total=temp;
  }
  return total;
}
  
const std::vector<float>& HcalNoiseHPD::allDigi(void) const
{
  return allDigi_;
}
  
float HcalNoiseHPD::allDigiTime(void) const
{
  float total=0, time=0;
  for(unsigned int i=0; i<allDigi_.size(); i++) {
    time  += i*allDigi_[i];
    total += allDigi_[i];
  }
  return total!=0.0 ? time/total : -999.;
}
  
float HcalNoiseHPD::allDigiTotal(void) const
{
  float total=0;
  for(unsigned int i=0; i<allDigi_.size(); i++) {
    total += allDigi_[i];
  }
  return total;
}
  
float HcalNoiseHPD::allDigiHighest2TS(void) const
{
  float total=0;
  for(unsigned int i=0; i<allDigi_.size()-1; i++) {
    float temp = allDigi_[i]+allDigi_[i+1];
    if(temp>total) total=temp;
  }
  return total;
}
  
float HcalNoiseHPD::allDigiHighest3TS(void) const
{
  float total=0;
  for(unsigned int i=0; i<allDigi_.size()-2; i++) {
    float temp = allDigi_[i]+allDigi_[i+1]+allDigi_[i+2];
    if(temp>total) total=temp;
  }
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

const edm::RefVector<HBHERecHitCollection>& HcalNoiseHPD::recHits(void) const
{
  return rechits_;
}
  
float HcalNoiseHPD::recHitEnergy(float threshold) const
{
  float total=0.0;
  for(edm::RefVector<HBHERecHitCollection>::const_iterator it=rechits_.begin(); it!=rechits_.end(); ++it) {
    float energy=(*it)->energy();
    if(energy>=threshold) total+=energy;
  }
  return total;
}
  
float HcalNoiseHPD::minRecHitTime(float threshold) const
{
  float mintime=9999999;
  for(edm::RefVector<HBHERecHitCollection>::const_iterator it=rechits_.begin(); it!=rechits_.end(); ++it) {
    if((*it)->energy()<threshold) continue;
    float time=(*it)->time();
    if(mintime>time) mintime=time;
  }
  return mintime;
}
  
float HcalNoiseHPD::maxRecHitTime(float threshold) const
{
  float maxtime=-9999999;
  for(edm::RefVector<HBHERecHitCollection>::const_iterator it=rechits_.begin(); it!=rechits_.end(); ++it) {
    if((*it)->energy()<threshold) continue;
    float time=(*it)->time();
    if(maxtime<time) maxtime=time;
  }
  return maxtime;
}

int HcalNoiseHPD::numRecHits(float threshold) const
{
  int count=0;
  for(edm::RefVector<HBHERecHitCollection>::const_iterator it=rechits_.begin(); it!=rechits_.end(); ++it)
    if((*it)->energy()>=threshold) ++count;
  return count;
}

const edm::RefVector<CaloTowerCollection>& HcalNoiseHPD::caloTowers(void) const
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
  return (e+h)!=0 ? e/(e+h) : -999.;
}

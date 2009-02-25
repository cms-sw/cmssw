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
    minTime_(999999.), maxTime_(-999999.), rechitEnergy_(0.0),
    numHits_(0), numHitsAboveThreshold_(0),
    twrHadE_(0.0), twrEmE_(0.0)
{
  for(int i=0; i<HBHEDataFrame::MAXSAMPLES; i++)
    bigDigi_[i]=big5Digi_[i]=allDigi_[i]=0.0;
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
  
  
DigiArray HcalNoiseHPD::bigDigi(void) const
{
  return bigDigi_;
}
  
DigiArray::const_iterator HcalNoiseHPD::beginBigDigi(void) const
{
  return bigDigi_.begin();
}
  
DigiArray::const_iterator HcalNoiseHPD::endBigDigi(void) const
{
  return bigDigi_.end();
}
  
double HcalNoiseHPD::bigDigiTime(void) const
{
  double total=0, time=0;
  for(unsigned int i=0; i<bigDigi_.size(); i++) {
    time  += i*bigDigi_[i];
    total += bigDigi_[i];
  }
  return total!=0.0 ? time/total : -999.;
}
  
double HcalNoiseHPD::bigDigiTotal(void) const
{
  double total=0;
  for(unsigned int i=0; i<bigDigi_.size(); i++) {
    total += bigDigi_[i];
  }
  return total;
}
  
double HcalNoiseHPD::bigDigiHighest2TS(void) const
{
  double total=0;
  for(unsigned int i=0; i<bigDigi_.size()-1; i++) {
    double temp = bigDigi_[i]+bigDigi_[i+1];
    if(temp>total) total=temp;
  }
  return total;
}
  
double HcalNoiseHPD::bigDigiHighest3TS(void) const
{
  double total=0;
  for(unsigned int i=0; i<bigDigi_.size()-2; i++) {
    double temp = bigDigi_[i]+bigDigi_[i+1]+bigDigi_[i+2];
    if(temp>total) total=temp;
  }
  return total;
}
  
DigiArray HcalNoiseHPD::big5Digi(void) const
{
  return big5Digi_;
}
  
DigiArray::const_iterator HcalNoiseHPD::beginBig5Digi(void) const
{
  return big5Digi_.begin();
}
  
DigiArray::const_iterator HcalNoiseHPD::endBig5Digi(void) const
{
  return big5Digi_.end();
}
  
double HcalNoiseHPD::big5DigiTime(void) const
{
  double total=0, time=0;
  for(unsigned int i=0; i<big5Digi_.size(); i++) {
    time  += i*big5Digi_[i];
    total += big5Digi_[i];
  }
  return total!=0.0 ? time/total : -999.;
}
  
double HcalNoiseHPD::big5DigiTotal(void) const
{
  double total=0;
  for(unsigned int i=0; i<big5Digi_.size(); i++) {
    total += big5Digi_[i];
  }
  return total;
}
  
double HcalNoiseHPD::big5DigiHighest2TS(void) const
{
  double total=0;
  for(unsigned int i=0; i<big5Digi_.size()-1; i++) {
    double temp = big5Digi_[i]+big5Digi_[i+1];
    if(temp>total) total=temp;
  }
  return total;
}
  
double HcalNoiseHPD::big5DigiHighest3TS(void) const
{
  double total=0;
  for(unsigned int i=0; i<big5Digi_.size()-2; i++) {
    double temp = big5Digi_[i]+big5Digi_[i+1]+big5Digi_[i+2];
    if(temp>total) total=temp;
  }
  return total;
}
  
DigiArray HcalNoiseHPD::allDigi(void) const
{
  return allDigi_;
}
  
DigiArray::const_iterator HcalNoiseHPD::beginAllDigi(void) const
{
  return allDigi_.begin();
}
  
DigiArray::const_iterator HcalNoiseHPD::endAllDigi(void) const
{
  return allDigi_.end();
}
  
double HcalNoiseHPD::allDigiTime(void) const
{
  double total=0, time=0;
  for(unsigned int i=0; i<allDigi_.size(); i++) {
    time  += i*allDigi_[i];
    total += allDigi_[i];
  }
  return total!=0.0 ? time/total : -999.;
}
  
double HcalNoiseHPD::allDigiTotal(void) const
{
  double total=0;
  for(unsigned int i=0; i<allDigi_.size(); i++) {
    total += allDigi_[i];
  }
  return total;
}
  
double HcalNoiseHPD::allDigiHighest2TS(void) const
{
  double total=0;
  for(unsigned int i=0; i<allDigi_.size()-1; i++) {
    double temp = allDigi_[i]+allDigi_[i+1];
    if(temp>total) total=temp;
  }
  return total;
}
  
double HcalNoiseHPD::allDigiHighest3TS(void) const
{
  double total=0;
  for(unsigned int i=0; i<allDigi_.size()-2; i++) {
    double temp = allDigi_[i]+allDigi_[i+1]+allDigi_[i+2];
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
  
double HcalNoiseHPD::minTime(void) const
{
  return minTime_;
}
  
double HcalNoiseHPD::maxTime(void) const
{
  return maxTime_;
}
  
double HcalNoiseHPD::rechitEnergy(void) const
{
  return rechitEnergy_;
}
  
int HcalNoiseHPD::numHits(void) const
{
  return numHits_;
}
  
int HcalNoiseHPD::numHitsAboveThreshold(void) const
{
  return numHitsAboveThreshold_;
}
  
double HcalNoiseHPD::caloTowerHadE(void) const
{
  return twrHadE_;
}
  
double HcalNoiseHPD::caloTowerEmE(void) const
{
  return twrEmE_;
}

double HcalNoiseHPD::caloTowerTotalE(void) const
{
  return twrEmE_+twrHadE_;
}
  
double HcalNoiseHPD::caloTowerEmFraction(void) const
{
  return caloTowerTotalE()!=0.0 ? twrEmE_/caloTowerTotalE() : -999.;
}

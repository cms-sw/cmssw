//
// HcalNoiseSummary.cc
//
//    description: implementation of container class of HCAL noise summary
//
//    author: J.P. Chou, Brown
//


#include "DataFormats/METReco/interface/HcalNoiseSummary.h"

// default constructor
HcalNoiseSummary::HcalNoiseSummary()
  : filterstatus_(0), emenergy_(0.0), hadenergy_(0.0), trackenergy_(0.0),
    min10_(0.0), max10_(0.0), rms10_(0.0),
    min25_(0.0), max25_(0.0), rms25_(0.0),
    nproblemRBXs_(0)
{
}

// destructor
HcalNoiseSummary::~HcalNoiseSummary()
{
}

// accessors
bool HcalNoiseSummary::passNoiseFilter(void) const
{
  return (filterstatus_==0);
}

int HcalNoiseSummary::noiseFilterStatus(void) const
{
  return filterstatus_;
}

float HcalNoiseSummary::eventEMEnergy(void) const
{
  return emenergy_;
}

float HcalNoiseSummary::eventHadEnergy(void) const
{
  return hadenergy_;
}

float HcalNoiseSummary::eventTrackEnergy(void) const
{
  return trackenergy_;
}

float HcalNoiseSummary::eventEMFraction(void) const
{
  if(hadenergy_+emenergy_==0.0) return -999.;
  else return emenergy_/(hadenergy_+emenergy_);
}

float HcalNoiseSummary::eventChargeFraction(void) const
{
  if(hadenergy_+emenergy_==0.0) return -999.;
  else return trackenergy_/(hadenergy_+emenergy_);
}

float HcalNoiseSummary::min10GeVHitTime(void) const
{
  return min10_;
}

float HcalNoiseSummary::max10GeVHitTime(void) const
{
  return max10_;
}

float HcalNoiseSummary::rms10GeVHitTime(void) const
{
  return rms10_;
}

float HcalNoiseSummary::min25GeVHitTime(void) const
{
  return min25_;
}
 
float HcalNoiseSummary::max25GeVHitTime(void) const
{
  return max25_;
}
 
float HcalNoiseSummary::rms25GeVHitTime(void) const
{
  return rms25_;
}
  
int HcalNoiseSummary::numProblematicRBXs(void) const
{
  return nproblemRBXs_;
}

edm::RefVector<reco::CaloJetCollection> HcalNoiseSummary::problematicJets(void) const
{
  return problemjets_;
}

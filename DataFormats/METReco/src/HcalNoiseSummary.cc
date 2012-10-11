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
  : filterstatus_(0), noisetype_(0), emenergy_(0.0), hadenergy_(0.0), trackenergy_(0.0),
    min10_(999999.), max10_(-999999.), rms10_(0.0),
    min25_(999999.), max25_(-999999.), rms25_(0.0),
    cnthit10_(0), cnthit25_(0),
    mine2ts_(0.), mine10ts_(0.),
    maxe2ts_(0.), maxe10ts_(0.),
    maxzeros_(0),
    maxhpdhits_(0), maxhpdhitsnoother_(0), maxrbxhits_(0),
    minhpdemf_(999999.), minrbxemf_(999999.),
    nproblemRBXs_(0),
    nisolnoise_(0), isolnoisee_(0), isolnoiseet_(0),
    nflatnoise_(0), flatnoisee_(0), flatnoiseet_(0),
    nspikenoise_(0), spikenoisee_(0), spikenoiseet_(0),
    ntrianglenoise_(0), trianglenoisee_(0), trianglenoiseet_(0),
    nts4ts5noise_(0), ts4ts5noisee_(0), ts4ts5noiseet_(0),
    rechitCount_(0), rechitCount15_(0), rechitEnergy_(0), rechitEnergy15_(0), calibCharge_(0),
    hasBadRBXTS4TS5_(false),
    calibCountTS45_(0),
    calibCountgt15TS45_(0),
    calibChargeTS45_(0.),
    calibChargegt15TS45_(0.),
    calibCountHF_(0),
    calibChargeHF_(0.),
    hitsInLaserRegion_(0),
    hitsInNonLaserRegion_(0),
    energyInLaserRegion_(0.),
    energyInNonLaserRegion_(0.)
{
}

// destructor
HcalNoiseSummary::~HcalNoiseSummary()
{
}

// accessors
bool HcalNoiseSummary::passLooseNoiseFilter(void) const
{
  return (filterstatus_ & 0xFF)==0;
}

bool HcalNoiseSummary::passTightNoiseFilter(void) const
{
  return (filterstatus_ & 0xFF00)==0;
}

bool HcalNoiseSummary::passHighLevelNoiseFilter(void) const
{
  return (filterstatus_ & 0xFF0000)==0;
}

int HcalNoiseSummary::noiseType(void) const
{
  if(maxRBXHits()>18) return 3;
  else if(maxRBXHits()>8) return 2;
  return 1;
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
  return cnthit10_>0 ? std::sqrt(rms10_/cnthit10_) : 999;
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
  return cnthit25_>0 ? std::sqrt(rms25_/cnthit25_) : 999;
}

int HcalNoiseSummary::num10GeVHits(void) const
{
  return cnthit10_;
}

int HcalNoiseSummary::num25GeVHits(void) const
{
  return cnthit25_;
}

float HcalNoiseSummary::minE2TS(void) const
{
  return mine2ts_;
}

float HcalNoiseSummary::minE10TS(void) const
{
  return mine10ts_;
}

float HcalNoiseSummary::minE2Over10TS(void) const
{
  return mine10ts_==0 ? 999999. : mine2ts_/mine10ts_;
}

float HcalNoiseSummary::maxE2TS(void) const
{
  return maxe2ts_;
}

float HcalNoiseSummary::maxE10TS(void) const
{
  return maxe10ts_;
}

float HcalNoiseSummary::maxE2Over10TS(void) const
{
  return maxe10ts_==0 ? -999999. : maxe2ts_/maxe10ts_;
}

int HcalNoiseSummary::maxZeros(void) const
{
  return maxzeros_;
}

int HcalNoiseSummary::maxHPDHits(void) const
{
  return maxhpdhits_;
}

int HcalNoiseSummary::maxHPDNoOtherHits(void) const
{
  return maxhpdhitsnoother_;
}

int HcalNoiseSummary::maxRBXHits(void) const
{
  return maxrbxhits_;
}

float HcalNoiseSummary::minHPDEMF(void) const
{
  return minhpdemf_;
}

float HcalNoiseSummary::minRBXEMF(void) const
{
  return minrbxemf_;
}

int HcalNoiseSummary::numProblematicRBXs(void) const
{
  return nproblemRBXs_;
}

int HcalNoiseSummary::numIsolatedNoiseChannels(void) const
{
  return nisolnoise_;
}

float HcalNoiseSummary::isolatedNoiseSumE(void) const
{
  return isolnoisee_;
}

float HcalNoiseSummary::isolatedNoiseSumEt(void) const
{
  return isolnoiseet_;
}

int HcalNoiseSummary::numFlatNoiseChannels(void) const
{
  return nflatnoise_;
}

float HcalNoiseSummary::flatNoiseSumE(void) const
{
  return flatnoisee_;
}

float HcalNoiseSummary::flatNoiseSumEt(void) const
{
  return flatnoiseet_;
}

int HcalNoiseSummary::numSpikeNoiseChannels(void) const
{
  return nspikenoise_;
}

float HcalNoiseSummary::spikeNoiseSumE(void) const
{
  return spikenoisee_;
}

float HcalNoiseSummary::spikeNoiseSumEt(void) const
{
  return spikenoiseet_;
}

int HcalNoiseSummary::numTriangleNoiseChannels(void) const
{
  return ntrianglenoise_;
}

float HcalNoiseSummary::triangleNoiseSumE(void) const
{
  return trianglenoisee_;
}

float HcalNoiseSummary::triangleNoiseSumEt(void) const
{
  return trianglenoiseet_;
}

int HcalNoiseSummary::numTS4TS5NoiseChannels(void) const
{
  return nts4ts5noise_;
}

float HcalNoiseSummary::TS4TS5NoiseSumE(void) const
{
  return ts4ts5noisee_;
}

float HcalNoiseSummary::TS4TS5NoiseSumEt(void) const
{
  return ts4ts5noiseet_;
}

int HcalNoiseSummary::GetRecHitCount(void) const
{
   return rechitCount_;
}

int HcalNoiseSummary::GetRecHitCount15(void) const
{
   return rechitCount15_;
}

double HcalNoiseSummary::GetRecHitEnergy(void) const
{
   return rechitEnergy_;
}

double HcalNoiseSummary::GetRecHitEnergy15(void) const
{
   return rechitEnergy15_;
}

double HcalNoiseSummary::GetTotalCalibCharge(void) const
{
   return calibCharge_;
}

bool HcalNoiseSummary::HasBadRBXTS4TS5(void) const
{
   return hasBadRBXTS4TS5_;
}

int HcalNoiseSummary::GetCalibCountTS45(void) const
{
  return calibCountTS45_;
}

int HcalNoiseSummary::GetCalibgt15CountTS45(void) const
{
  return calibCountgt15TS45_;
}

double HcalNoiseSummary::GetCalibChargeTS45(void) const
{
  return calibChargeTS45_;
}

double HcalNoiseSummary::GetCalibgt15ChargeTS45(void) const
{
  return calibChargegt15TS45_;
}

int HcalNoiseSummary::GetCalibCountHF(void) const
{
  // calibCountHF_ is the number of HF calibration digis in the event (no charge threshold required when counting these digis)
  return calibCountHF_;
}

double HcalNoiseSummary::GetCalibChargeHF(void) const
{
  // calibChargeHF_ is the total charge (over all 10 time slices) for all HF calibration digis in the event
  return calibChargeHF_;
}

int HcalNoiseSummary::GetHitsInNonLaserRegion(void) const
{
  return hitsInNonLaserRegion_;
}

double HcalNoiseSummary::GetEnergyInNonLaserRegion(void) const
{
  return energyInNonLaserRegion_;
}

int HcalNoiseSummary::GetHitsInLaserRegion(void) const
{
  return hitsInLaserRegion_;
}

double HcalNoiseSummary::GetEnergyInLaserRegion(void) const
{
  return energyInLaserRegion_;
}

edm::RefVector<reco::CaloJetCollection> HcalNoiseSummary::problematicJets(void) const
{
  return problemjets_;
}

edm::RefVector<CaloTowerCollection> HcalNoiseSummary::looseNoiseTowers(void) const
{
  return loosenoisetwrs_;
}

edm::RefVector<CaloTowerCollection> HcalNoiseSummary::tightNoiseTowers(void) const
{
  return tightnoisetwrs_;
}

edm::RefVector<CaloTowerCollection> HcalNoiseSummary::highLevelNoiseTowers(void) const
{
  return hlnoisetwrs_;
}

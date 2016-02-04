#include "RecoMET/METAlgorithms/interface/HcalNoiseAlgo.h"

CommonHcalNoiseRBXData::CommonHcalNoiseRBXData(const reco::HcalNoiseRBX& rbx, double minRecHitE, double minLowHitE, double minHighHitE)
{
  // energy
  energy_ = rbx.recHitEnergy(minRecHitE); 

  // ratio
  e2ts_ = rbx.allChargeHighest2TS();
  e10ts_ = rbx.allChargeTotal();

  // # of hits
  numHPDHits_ = 0;
  for(std::vector<reco::HcalNoiseHPD>::const_iterator it1=rbx.HPDsBegin(); it1!=rbx.HPDsEnd(); ++it1) {
    int nhpdhits=it1->numRecHits(minRecHitE);
    if(numHPDHits_ < nhpdhits) numHPDHits_ = nhpdhits;
  }
  numRBXHits_ = rbx.numRecHits(minRecHitE);
  numHPDNoOtherHits_ = (numHPDHits_ == numRBXHits_) ? numHPDHits_ : 0;    

  // # of ADC zeros
  numZeros_ = rbx.totalZeros();

  // timing
  minLowEHitTime_ = minHighEHitTime_ = 99999.;
  maxLowEHitTime_ = maxHighEHitTime_ = -99999.;
  lowEHitTimeSqrd_ = highEHitTimeSqrd_ = 0;
  numLowEHits_ = numHighEHits_ = 0;
  for(std::vector<reco::HcalNoiseHPD>::const_iterator it1=rbx.HPDsBegin(); it1!=rbx.HPDsEnd(); ++it1) {
    edm::RefVector<HBHERecHitCollection> rechits=it1->recHits();
    for(edm::RefVector<HBHERecHitCollection>::const_iterator it2=rechits.begin(); it2!=rechits.end(); ++it2) {
      float energy=(*it2)->energy();
      float time=(*it2)->time();
      if(energy>=minLowHitE) {
	if(minLowEHitTime_ > time) minLowEHitTime_ = time;
	if(maxLowEHitTime_ < time) maxLowEHitTime_ = time;
	lowEHitTimeSqrd_ += time*time;
	++numLowEHits_;
      }
      if(energy>=minHighHitE) {
	if(minHighEHitTime_ > time) minHighEHitTime_ = time;
	if(maxHighEHitTime_ < time) maxHighEHitTime_ = time;
	highEHitTimeSqrd_ += time*time;
	++numHighEHits_;
      }
    }
  }

  // emf
  HPDEMF_ = 999.;
  for(std::vector<reco::HcalNoiseHPD>::const_iterator it1=rbx.HPDsBegin(); it1!=rbx.HPDsEnd(); ++it1) {
    double eme=it1->caloTowerEmE();
    double hade=it1->recHitEnergy(minRecHitE);
    double emf=(eme+hade)==0 ? 999 : eme/(eme+hade);
    if(HPDEMF_ > emf) emf = HPDEMF_;
  }
  double eme=rbx.caloTowerEmE();
  RBXEMF_ = (eme+energy_)==0 ? 999 : eme/(eme+energy_);

  // calotowers
  rbxtowers_.clear();
  JoinCaloTowerRefVectorsWithoutDuplicates join;
  for(std::vector<reco::HcalNoiseHPD>::const_iterator it1=rbx.HPDsBegin(); it1!=rbx.HPDsEnd(); ++it1) {
    join(rbxtowers_, it1->caloTowers());
  }

  return;
}

HcalNoiseAlgo::HcalNoiseAlgo(const edm::ParameterSet& iConfig)
{
  pMinERatio_ = iConfig.getParameter<double>("pMinERatio");
  pMinEZeros_ = iConfig.getParameter<double>("pMinEZeros");
  pMinEEMF_ = iConfig.getParameter<double>("pMinEEMF");

  minERatio_ = iConfig.getParameter<double>("minERatio");
  minEZeros_ = iConfig.getParameter<double>("minEZeros");
  minEEMF_ = iConfig.getParameter<double>("minEEMF");  

  pMinE_ = iConfig.getParameter<double>("pMinE");
  pMinRatio_ = iConfig.getParameter<double>("pMinRatio");
  pMaxRatio_ = iConfig.getParameter<double>("pMaxRatio");
  pMinHPDHits_ = iConfig.getParameter<int>("pMinHPDHits");
  pMinRBXHits_ = iConfig.getParameter<int>("pMinRBXHits");
  pMinHPDNoOtherHits_ = iConfig.getParameter<int>("pMinHPDNoOtherHits");
  pMinZeros_ = iConfig.getParameter<int>("pMinZeros");
  pMinLowEHitTime_ = iConfig.getParameter<double>("pMinLowEHitTime");
  pMaxLowEHitTime_ = iConfig.getParameter<double>("pMaxLowEHitTime");
  pMinHighEHitTime_ = iConfig.getParameter<double>("pMinHighEHitTime");
  pMaxHighEHitTime_ = iConfig.getParameter<double>("pMaxHighEHitTime");
  pMaxHPDEMF_ = iConfig.getParameter<double>("pMaxHPDEMF");
  pMaxRBXEMF_ = iConfig.getParameter<double>("pMaxRBXEMF");

  lMinRatio_ = iConfig.getParameter<double>("lMinRatio");
  lMaxRatio_ = iConfig.getParameter<double>("lMaxRatio");
  lMinHPDHits_ = iConfig.getParameter<int>("lMinHPDHits");
  lMinRBXHits_ = iConfig.getParameter<int>("lMinRBXHits");
  lMinHPDNoOtherHits_ = iConfig.getParameter<int>("lMinHPDNoOtherHits");
  lMinZeros_ = iConfig.getParameter<int>("lMinZeros");
  lMinLowEHitTime_ = iConfig.getParameter<double>("lMinLowEHitTime");
  lMaxLowEHitTime_ = iConfig.getParameter<double>("lMaxLowEHitTime");
  lMinHighEHitTime_ = iConfig.getParameter<double>("lMinHighEHitTime");
  lMaxHighEHitTime_ = iConfig.getParameter<double>("lMaxHighEHitTime");

  tMinRatio_ = iConfig.getParameter<double>("tMinRatio");
  tMaxRatio_ = iConfig.getParameter<double>("tMaxRatio");
  tMinHPDHits_ = iConfig.getParameter<int>("tMinHPDHits");
  tMinRBXHits_ = iConfig.getParameter<int>("tMinRBXHits");
  tMinHPDNoOtherHits_ = iConfig.getParameter<int>("tMinHPDNoOtherHits");
  tMinZeros_ = iConfig.getParameter<int>("tMinZeros");
  tMinLowEHitTime_ = iConfig.getParameter<double>("tMinLowEHitTime");
  tMaxLowEHitTime_ = iConfig.getParameter<double>("tMaxLowEHitTime");
  tMinHighEHitTime_ = iConfig.getParameter<double>("tMinHighEHitTime");
  tMaxHighEHitTime_ = iConfig.getParameter<double>("tMaxHighEHitTime");

  hlMaxHPDEMF_ = iConfig.getParameter<double>("hlMaxHPDEMF");
  hlMaxRBXEMF_ = iConfig.getParameter<double>("hlMaxRBXEMF");
}

bool HcalNoiseAlgo::isProblematic(const CommonHcalNoiseRBXData& data) const
{
  if(data.energy()>pMinE_) return true;
  if(data.validRatio() && data.energy()>pMinERatio_ && data.ratio()<pMinRatio_) return true;
  if(data.validRatio() && data.energy()>pMinERatio_ && data.ratio()>pMaxRatio_) return true;
  if(data.numHPDHits()>=pMinHPDHits_) return true;
  if(data.numRBXHits()>=pMinRBXHits_) return true;
  if(data.numHPDNoOtherHits()>=pMinHPDNoOtherHits_) return true;
  if(data.numZeros()>=pMinZeros_ && data.energy()>pMinEZeros_) return true;
  if(data.minLowEHitTime()<pMinLowEHitTime_) return true;
  if(data.maxLowEHitTime()>pMaxLowEHitTime_) return true;
  if(data.minHighEHitTime()<pMinHighEHitTime_) return true;
  if(data.maxHighEHitTime()>pMaxHighEHitTime_) return true;
  if(data.HPDEMF()<pMaxHPDEMF_ && data.energy()>pMinEEMF_) return true;
  if(data.RBXEMF()<pMaxRBXEMF_ && data.energy()>pMinEEMF_) return true;  return false;
}


bool HcalNoiseAlgo::passLooseNoiseFilter(const CommonHcalNoiseRBXData& data) const
{
  return (passLooseRatio(data) && passLooseHits(data) && passLooseZeros(data) && passLooseTiming(data));
}

bool HcalNoiseAlgo::passTightNoiseFilter(const CommonHcalNoiseRBXData& data) const
{
  return (passTightRatio(data) && passTightHits(data) && passTightZeros(data) && passTightTiming(data));
}

bool HcalNoiseAlgo::passHighLevelNoiseFilter(const CommonHcalNoiseRBXData& data) const
{
  if(passEMFThreshold(data)) {
    if(data.HPDEMF()<hlMaxHPDEMF_) return false;
    if(data.RBXEMF()<hlMaxRBXEMF_) return false;
  }
  return true;
}

bool HcalNoiseAlgo::passLooseRatio(const CommonHcalNoiseRBXData& data) const
{
  if(passRatioThreshold(data)) {
    if(data.validRatio() && data.ratio()<lMinRatio_) return false;
    if(data.validRatio() && data.ratio()>lMaxRatio_) return false;
  }
  return true;
}

bool HcalNoiseAlgo::passLooseHits(const CommonHcalNoiseRBXData& data) const
{
  if(data.numHPDHits()>=lMinHPDHits_) return false;
  if(data.numRBXHits()>=lMinRBXHits_) return false;
  if(data.numHPDNoOtherHits()>=lMinHPDNoOtherHits_) return false;
  return true;
}

bool HcalNoiseAlgo::passLooseZeros(const CommonHcalNoiseRBXData& data) const
{
  if(passZerosThreshold(data)) {
    if(data.numZeros()>=lMinZeros_) return false;
  }
  return true;
}

bool HcalNoiseAlgo::passLooseTiming(const CommonHcalNoiseRBXData& data) const
{
  if(data.minLowEHitTime()<lMinLowEHitTime_) return false;
  if(data.maxLowEHitTime()>lMaxLowEHitTime_) return false;
  if(data.minHighEHitTime()<lMinHighEHitTime_) return false;
  if(data.maxHighEHitTime()>lMaxHighEHitTime_) return false;
  return true;
}

bool HcalNoiseAlgo::passTightRatio(const CommonHcalNoiseRBXData& data) const
{
  if(passRatioThreshold(data)) {
    if(data.validRatio() && data.ratio()<tMinRatio_) return false;
    if(data.validRatio() && data.ratio()>tMaxRatio_) return false;
  }
  return true;
}

bool HcalNoiseAlgo::passTightHits(const CommonHcalNoiseRBXData& data) const
{
  if(data.numHPDHits()>=tMinHPDHits_) return false;
  if(data.numRBXHits()>=tMinRBXHits_) return false;
  if(data.numHPDNoOtherHits()>=tMinHPDNoOtherHits_) return false;
  return true;
}

bool HcalNoiseAlgo::passTightZeros(const CommonHcalNoiseRBXData& data) const
{
  if(passZerosThreshold(data)) {
    if(data.numZeros()>=tMinZeros_) return false;
  }
  return true;
}

bool HcalNoiseAlgo::passTightTiming(const CommonHcalNoiseRBXData& data) const
{
  if(data.minLowEHitTime()<tMinLowEHitTime_) return false;
  if(data.maxLowEHitTime()>tMaxLowEHitTime_) return false;
  if(data.minHighEHitTime()<tMinHighEHitTime_) return false;
  if(data.maxHighEHitTime()>tMaxHighEHitTime_) return false;
  return true;
}

bool HcalNoiseAlgo::passRatioThreshold(const CommonHcalNoiseRBXData& data) const
{
  return (data.energy()>minERatio_);
}

bool HcalNoiseAlgo::passZerosThreshold(const CommonHcalNoiseRBXData& data) const
{
  return (data.energy()>minEZeros_);
}

bool HcalNoiseAlgo::passEMFThreshold(const CommonHcalNoiseRBXData& data) const
{
  return (data.energy()>minEEMF_);
}

void JoinCaloTowerRefVectorsWithoutDuplicates::operator()(edm::RefVector<CaloTowerCollection>& v1, const edm::RefVector<CaloTowerCollection>& v2) const
{
  // combines them first into a set to get rid of duplicates and then puts them into the first vector

  // sorts them first to get rid of duplicates, then puts them into another RefVector
  twrrefset_t twrrefset;
  for(edm::RefVector<CaloTowerCollection>::const_iterator it=v1.begin(); it!=v1.end(); ++it)
    twrrefset.insert(*it);
  for(edm::RefVector<CaloTowerCollection>::const_iterator it=v2.begin(); it!=v2.end(); ++it)
    twrrefset.insert(*it);

  // clear the original refvector and put them back in
  v1.clear();
  for(twrrefset_t::const_iterator it=twrrefset.begin(); it!=twrrefset.end(); ++it) {
    v1.push_back(*it);
  }
  return;
}

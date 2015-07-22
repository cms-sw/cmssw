#include "RecoMET/METAlgorithms/interface/HcalNoiseAlgo.h"

CommonHcalNoiseRBXData::CommonHcalNoiseRBXData(const reco::HcalNoiseRBX& rbx, double minRecHitE,
   double minLowHitE, double minHighHitE, double TS4TS5EnergyThreshold,
   std::vector<std::pair<double, double> > &TS4TS5UpperCut,
   std::vector<std::pair<double, double> > &TS4TS5LowerCut,
   double minRBXRechitR45E)
  :r45Count_(0)
  ,r45Fraction_(0)
  ,r45EnergyFraction_(0)
{
  // energy
  energy_ = rbx.recHitEnergy(minRecHitE); 

  // ratio
  e2ts_ = rbx.allChargeHighest2TS();
  e10ts_ = rbx.allChargeTotal();

  // TS4TS5
  TS4TS5Decision_ = true;
  if(energy_ > TS4TS5EnergyThreshold)   // check filter
  {
     std::vector<float> AllCharge = rbx.allCharge();
     double BaseCharge = AllCharge[4] + AllCharge[5];
     if(BaseCharge < 1)
        BaseCharge = 1;
     double TS4TS5 = (AllCharge[4] - AllCharge[5]) / BaseCharge;

     if(CheckPassFilter(BaseCharge, TS4TS5, TS4TS5UpperCut, 1) == false)
        TS4TS5Decision_ = false;
     if(CheckPassFilter(BaseCharge, TS4TS5, TS4TS5LowerCut, -1) == false)
        TS4TS5Decision_ = false;
  }
  else
     TS4TS5Decision_ = true;

  // Rechit-wide R45
  int rbxHitCount = rbx.numRecHits(minRBXRechitR45E);
  if(rbxHitCount > 0)
  {
     r45Count_ = rbx.numRecHitsFailR45(minRBXRechitR45E);
     r45Fraction_ = r45Count_ / rbxHitCount;
     r45EnergyFraction_ = rbx.recHitEnergyFailR45(minRBXRechitR45E) / rbx.recHitEnergy(minRBXRechitR45E);
  }

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

  if(iConfig.existsAs<int>("pMinRBXRechitR45Count"))
     pMinRBXRechitR45Count_ = iConfig.getParameter<int>("pMinRBXRechitR45Count");
  else
     pMinRBXRechitR45Count_ = 0;
  if(iConfig.existsAs<double>("pMinRBXRechitR45Fraction"))
     pMinRBXRechitR45Fraction_ = iConfig.getParameter<double>("pMinRBXRechitR45Fraction");
  else
     pMinRBXRechitR45Fraction_ = 0;
  if(iConfig.existsAs<double>("pMinRechitR45EnergyFraction"))
     pMinRBXRechitR45EnergyFraction_ = iConfig.getParameter<double>("pMinRBXRechitR45EnergyFraction");
  else
     pMinRBXRechitR45EnergyFraction_ = 0;

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

  if(iConfig.existsAs<std::vector<double> >("lRBXRecHitR45Cuts"))
     lMinRBXRechitR45Cuts_ = iConfig.getParameter<std::vector<double> >("lRBXRecHitR45Cuts");
  else
  {
     double defaultCut[4] = {-999, -999, -999, -999};
     lMinRBXRechitR45Cuts_ = std::vector<double>(defaultCut, defaultCut + 4);
  }

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

  if(iConfig.existsAs<std::vector<double> >("tRBXRecHitR45Cuts"))
     tMinRBXRechitR45Cuts_ = iConfig.getParameter<std::vector<double> >("tRBXRecHitR45Cuts");
  else
  {
     double defaultCut[4] = {-999, -999, -999, -999};
     tMinRBXRechitR45Cuts_ = std::vector<double>(defaultCut, defaultCut + 4);
  }

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
  if(data.RBXEMF()<pMaxRBXEMF_ && data.energy()>pMinEEMF_) return true;
  if(data.r45Count() >= pMinRBXRechitR45Count_)   return true;
  if(data.r45Fraction() >= pMinRBXRechitR45Fraction_)   return true;
  if(data.r45EnergyFraction() >= pMinRBXRechitR45EnergyFraction_)   return true;
  
  return false;
}


bool HcalNoiseAlgo::passLooseNoiseFilter(const CommonHcalNoiseRBXData& data) const
{
  return (passLooseRatio(data) && passLooseHits(data) && passLooseZeros(data) && passLooseTiming(data) && passLooseRBXRechitR45(data));
}

bool HcalNoiseAlgo::passTightNoiseFilter(const CommonHcalNoiseRBXData& data) const
{
  return (passTightRatio(data) && passTightHits(data) && passTightZeros(data) && passTightTiming(data) && passTightRBXRechitR45(data));
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

bool HcalNoiseAlgo::passLooseRBXRechitR45(const CommonHcalNoiseRBXData &data) const
{
   int Count = data.r45Count();
   double Fraction = data.r45Fraction();
   double EnergyFraction = data.r45EnergyFraction();

   for(int i = 0; i + 3 < (int)lMinRBXRechitR45Cuts_.size(); i = i + 4)
   {
      double Value = Count * lMinRBXRechitR45Cuts_[i] + Fraction * lMinRBXRechitR45Cuts_[i+1] + EnergyFraction * lMinRBXRechitR45Cuts_[i+2] + lMinRBXRechitR45Cuts_[i+3];
      if(Value > 0)
         return false;
   }
   
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

bool HcalNoiseAlgo::passTightRBXRechitR45(const CommonHcalNoiseRBXData &data) const
{
   int Count = data.r45Count();
   double Fraction = data.r45Fraction();
   double EnergyFraction = data.r45EnergyFraction();

   for(int i = 0; i + 3 < (int)tMinRBXRechitR45Cuts_.size(); i = i + 4)
   {
      double Value = Count * tMinRBXRechitR45Cuts_[i] + Fraction * tMinRBXRechitR45Cuts_[i+1] + EnergyFraction * tMinRBXRechitR45Cuts_[i+2] + tMinRBXRechitR45Cuts_[i+3];
      if(Value > 0)
         return false;
   }
   
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

bool CommonHcalNoiseRBXData::CheckPassFilter(double Charge, double Discriminant,
   std::vector<std::pair<double, double> > &Cuts, int Side)
{
   //
   // Checks whether Discriminant value passes Cuts for the specified Charge.  True if pulse is good.
   //
   // The "Cuts" pairs are assumed to be sorted in terms of size from small to large,
   //    where each "pair" = (Charge, Discriminant)
   // "Side" is either positive or negative, which determines whether to discard the pulse if discriminant
   //    is greater or smaller than the cut value
   //

   if(Cuts.size() == 0)   // safety check that there are some cuts defined
      return true;

   if(Charge <= Cuts[0].first)   // too small to cut on
      return true;

   int IndexLargerThanCharge = -1;   // find the range it is falling in
   for(int i = 1; i < (int)Cuts.size(); i++)
   {
      if(Cuts[i].first > Charge)
      {
         IndexLargerThanCharge = i;
         break;
      }
   }

   double limit = 1000000;

   if(IndexLargerThanCharge == -1)   // if charge is greater than the last entry, assume flat line
      limit = Cuts[Cuts.size()-1].second;
   else   // otherwise, do a linear interpolation to find the cut position
   {
      double C1 = Cuts[IndexLargerThanCharge].first;
      double C2 = Cuts[IndexLargerThanCharge-1].first;
      double L1 = Cuts[IndexLargerThanCharge].second;
      double L2 = Cuts[IndexLargerThanCharge-1].second;

      limit = (Charge - C1) / (C2 - C1) * (L2 - L1) + L1;
   }

   if(Side > 0 && Discriminant > limit)
      return false;
   if(Side < 0 && Discriminant < limit)
      return false;

   return true;
}


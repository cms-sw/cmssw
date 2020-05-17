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
    : idnumber_(0),
      totalZeros_(0),
      maxZeros_(0),
      bigCharge_(HBHEDataFrame::MAXSAMPLES, 0.0),
      big5Charge_(HBHEDataFrame::MAXSAMPLES, 0.0) {
  // reserve some space, so that there's no reallocation issues
  rechits_.reserve(19);
  calotowers_.reserve(19);
}

// destructor
HcalNoiseHPD::~HcalNoiseHPD() {}

// accessors
int HcalNoiseHPD::idnumber(void) const { return idnumber_; }

const std::vector<float> HcalNoiseHPD::bigCharge(void) const { return bigCharge_; }

float HcalNoiseHPD::bigChargeTotal(void) const {
  float total = 0;
  for (float i : bigCharge_) {
    total += i;
  }
  return total;
}

float HcalNoiseHPD::bigChargeHighest2TS(unsigned int firstts) const {
  float total = 0;
  for (unsigned int i = firstts; i < firstts + 2 && i < bigCharge_.size(); i++)
    total += bigCharge_[i];
  return total;
}

float HcalNoiseHPD::bigChargeHighest3TS(unsigned int firstts) const {
  float total = 0;
  for (unsigned int i = firstts; i < firstts + 3 && i < bigCharge_.size(); i++)
    total += bigCharge_[i];
  return total;
}

const std::vector<float> HcalNoiseHPD::big5Charge(void) const { return big5Charge_; }

float HcalNoiseHPD::big5ChargeTotal(void) const {
  float total = 0;
  for (float i : big5Charge_) {
    total += i;
  }
  return total;
}

float HcalNoiseHPD::big5ChargeHighest2TS(unsigned int firstts) const {
  float total = 0;
  for (unsigned int i = firstts; i < firstts + 2 && i < big5Charge_.size(); i++)
    total += big5Charge_[i];
  return total;
}

float HcalNoiseHPD::big5ChargeHighest3TS(unsigned int firstts) const {
  float total = 0;
  for (unsigned int i = firstts; i < firstts + 2 && i < big5Charge_.size(); i++)
    total += big5Charge_[i];
  return total;
}

int HcalNoiseHPD::totalZeros(void) const { return totalZeros_; }

int HcalNoiseHPD::maxZeros(void) const { return maxZeros_; }

const edm::RefVector<HBHERecHitCollection> HcalNoiseHPD::recHits(void) const { return rechits_; }

float HcalNoiseHPD::recHitEnergy(const float threshold) const {
  double total = 0.0;
  for (auto&& rechit : rechits_) {
    const float energy = (rechit)->eraw();
    if (energy >= threshold)
      total += energy;
  }
  return total;
}

float HcalNoiseHPD::recHitEnergyFailR45(const float threshold) const {
  double total = 0.0;
  for (auto&& rechit : rechits_) {
    const float energy = (rechit)->eraw();
    if ((rechit)->flagField(HcalCaloFlagLabels::HBHETS4TS5Noise) && !(rechit)->flagField(HcalCaloFlagLabels::HBHEOOTPU))
      if (energy >= threshold)
        total += energy;
  }
  return total;
}

float HcalNoiseHPD::minRecHitTime(const float threshold) const {
  float mintime = 9999999;
  for (auto&& rechit : rechits_) {
    if ((rechit)->energy() < threshold)
      continue;
    float time = (rechit)->time();
    if (mintime > time)
      mintime = time;
  }
  return mintime;
}

float HcalNoiseHPD::maxRecHitTime(const float threshold) const {
  float maxtime = -9999999;
  for (auto&& rechit : rechits_) {
    if ((rechit)->energy() < threshold)
      continue;
    float time = (rechit)->time();
    if (maxtime < time)
      maxtime = time;
  }
  return maxtime;
}

int HcalNoiseHPD::numRecHits(const float threshold) const {
  int count = 0;
  for (auto&& rechit : rechits_) {
    // Exclude uncollapsed QIE11 channels
    if (CaloRecHitAuxSetter::getBit((rechit)->auxPhase1(), HBHERecHitAuxSetter::OFF_TDC_TIME) &&
        !CaloRecHitAuxSetter::getBit((rechit)->auxPhase1(), HBHERecHitAuxSetter::OFF_COMBINED))
      continue;
    if ((rechit)->eraw() >= threshold)
      ++count;
  }
  return count;
}

int HcalNoiseHPD::numRecHitsFailR45(const float threshold) const {
  int count = 0;
  for (auto&& rechit : rechits_)
    if ((rechit)->flagField(HcalCaloFlagLabels::HBHETS4TS5Noise) && !(rechit)->flagField(HcalCaloFlagLabels::HBHEOOTPU))
      if ((rechit)->eraw() >= threshold)
        ++count;
  return count;
}

const edm::RefVector<CaloTowerCollection> HcalNoiseHPD::caloTowers(void) const { return calotowers_; }

double HcalNoiseHPD::caloTowerHadE(void) const {
  double total = 0;
  for (auto&& calotower : calotowers_)
    total += (calotower)->hadEnergy();
  return total;
}

double HcalNoiseHPD::caloTowerEmE(void) const {
  double total = 0;
  for (auto&& calotower : calotowers_)
    total += (calotower)->emEnergy();
  return total;
}

double HcalNoiseHPD::caloTowerTotalE(void) const {
  double total = 0;
  for (auto&& calotower : calotowers_)
    total += (calotower)->emEnergy() + (calotower)->hadEnergy();
  return total;
}

double HcalNoiseHPD::caloTowerEmFraction(void) const {
  double h = 0, e = 0;
  for (auto&& calotower : calotowers_) {
    e += (calotower)->emEnergy();
    h += (calotower)->hadEnergy();
  }
  return (e + h) != 0 ? e / (e + h) : 999.;
}

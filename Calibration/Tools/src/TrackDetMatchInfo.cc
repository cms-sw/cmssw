#include <algorithm>
#include "Calibration/Tools/interface/TrackDetMatchInfo.h"

double HTrackDetMatchInfo::ecalEnergyFromRecHits() {
  double energy(0);
  for (auto hit = crossedEcalRecHits.begin(); hit != crossedEcalRecHits.end(); hit++)
    energy += hit->energy();
  return energy;
}

double HTrackDetMatchInfo::ecalConeEnergyFromRecHits() {
  double energy(0);
  for (auto hit = coneEcalRecHits.begin(); hit != coneEcalRecHits.end(); hit++) {
    energy += hit->energy();
    //     std::cout<< hit->detid().rawId()<<" "<<hit->energy()<<" "<<energy<<std::endl;
  }
  return energy;
}

double HTrackDetMatchInfo::ecalEnergyFromCaloTowers() {
  double energy(0);
  for (auto hit = crossedTowers.begin(); hit != crossedTowers.end(); hit++) {
    energy += hit->emEnergy();
  }
  return energy;
}

double HTrackDetMatchInfo::ecalConeEnergyFromCaloTowers() {
  double energy(0);
  for (auto hit = coneTowers.begin(); hit != coneTowers.end(); hit++)
    energy += hit->emEnergy();
  return energy;
}

double HTrackDetMatchInfo::hcalEnergyFromRecHits() {
  double energy(0);
  for (auto hit = crossedHcalRecHits.begin(); hit != crossedHcalRecHits.end(); hit++)
    energy += hit->energy();
  return energy;
}

double HTrackDetMatchInfo::hcalConeEnergyFromRecHits() {
  double energy(0);
  for (auto hit = coneHcalRecHits.begin(); hit != coneHcalRecHits.end(); hit++) {
    energy += hit->energy();
  }
  return energy;
}

double HTrackDetMatchInfo::hcalBoxEnergyFromRecHits() {
  double energy(0);
  for (auto hit = boxHcalRecHits.begin(); hit != boxHcalRecHits.end(); hit++)
    energy += hit->energy();
  return energy;
}

double HTrackDetMatchInfo::hcalEnergyFromCaloTowers() {
  double energy(0);
  for (auto tower = crossedTowers.begin(); tower != crossedTowers.end(); tower++)
    energy += tower->hadEnergy();
  return energy;
}

double HTrackDetMatchInfo::hcalConeEnergyFromCaloTowers() {
  double energy(0);
  for (auto hit = coneTowers.begin(); hit != coneTowers.end(); hit++) {
    energy += hit->hadEnergy();
  }
  return energy;
}

double HTrackDetMatchInfo::hcalBoxEnergyFromCaloTowers() {
  double energy(0);
  for (auto hit = boxTowers.begin(); hit != boxTowers.end(); hit++)
    energy += hit->hadEnergy();
  return energy;
}

double HTrackDetMatchInfo::outerHcalEnergy() {
  double energy(0);
  for (auto tower = crossedTowers.begin(); tower != crossedTowers.end(); tower++)
    energy += tower->outerEnergy();
  return energy;
}

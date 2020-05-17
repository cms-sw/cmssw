#include <algorithm>
#include "Calibration/Tools/interface/TrackDetMatchInfo.h"

double HTrackDetMatchInfo::ecalEnergyFromRecHits() {
  double energy(0);
  for (const auto& crossedEcalRecHit : crossedEcalRecHits)
    energy += crossedEcalRecHit.energy();
  return energy;
}

double HTrackDetMatchInfo::ecalConeEnergyFromRecHits() {
  double energy(0);
  for (const auto& coneEcalRecHit : coneEcalRecHits) {
    energy += coneEcalRecHit.energy();
    //     std::cout<< hit->detid().rawId()<<" "<<hit->energy()<<" "<<energy<<std::endl;
  }
  return energy;
}

double HTrackDetMatchInfo::ecalEnergyFromCaloTowers() {
  double energy(0);
  for (const auto& crossedTower : crossedTowers) {
    energy += crossedTower.emEnergy();
  }
  return energy;
}

double HTrackDetMatchInfo::ecalConeEnergyFromCaloTowers() {
  double energy(0);
  for (const auto& coneTower : coneTowers)
    energy += coneTower.emEnergy();
  return energy;
}

double HTrackDetMatchInfo::hcalEnergyFromRecHits() {
  double energy(0);
  for (const auto& crossedHcalRecHit : crossedHcalRecHits)
    energy += crossedHcalRecHit.energy();
  return energy;
}

double HTrackDetMatchInfo::hcalConeEnergyFromRecHits() {
  double energy(0);
  for (const auto& coneHcalRecHit : coneHcalRecHits) {
    energy += coneHcalRecHit.energy();
  }
  return energy;
}

double HTrackDetMatchInfo::hcalBoxEnergyFromRecHits() {
  double energy(0);
  for (const auto& boxHcalRecHit : boxHcalRecHits)
    energy += boxHcalRecHit.energy();
  return energy;
}

double HTrackDetMatchInfo::hcalEnergyFromCaloTowers() {
  double energy(0);
  for (const auto& crossedTower : crossedTowers)
    energy += crossedTower.hadEnergy();
  return energy;
}

double HTrackDetMatchInfo::hcalConeEnergyFromCaloTowers() {
  double energy(0);
  for (const auto& coneTower : coneTowers) {
    energy += coneTower.hadEnergy();
  }
  return energy;
}

double HTrackDetMatchInfo::hcalBoxEnergyFromCaloTowers() {
  double energy(0);
  for (const auto& boxTower : boxTowers)
    energy += boxTower.hadEnergy();
  return energy;
}

double HTrackDetMatchInfo::outerHcalEnergy() {
  double energy(0);
  for (const auto& crossedTower : crossedTowers)
    energy += crossedTower.outerEnergy();
  return energy;
}

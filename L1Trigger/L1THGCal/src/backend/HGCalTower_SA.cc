#include "L1Trigger/L1THGCal/interface/backend/HGCalTower_SA.h"

using namespace l1thgcfirmware;

HGCalTower& HGCalTower::operator+=(const HGCalTower& tower) {
  etEm_ += tower.etEm();
  etHad_ += tower.etHad();

  return *this;
}

void HGCalTower::addEtEm(double et) { etEm_ += et; }

void HGCalTower::addEtHad(double et) { etHad_ += et; }

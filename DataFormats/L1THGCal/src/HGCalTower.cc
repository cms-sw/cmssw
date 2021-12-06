#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "FWCore/Utilities/interface/EDMException.h"

using l1t::HGCalTower;
using l1t::L1Candidate;

HGCalTower::HGCalTower(double etEm,
                       double etHad,
                       double eta,
                       double phi,
                       uint32_t id,
                       int hwpt,
                       int hweta,
                       int hwphi,
                       int qual,
                       int hwEtEm,
                       int hwEtHad,
                       int hwEtRatio)
    : L1Candidate(PolarLorentzVector(etEm + etHad, eta, phi, 0.), hwpt, hweta, hwphi, qual),
      etEm_(etEm),
      etHad_(etHad),
      id_(id),
      hwEtEm_(hwEtEm),
      hwEtHad_(hwEtHad),
      hwEtRatio_(hwEtRatio) {}

HGCalTower::~HGCalTower() {}

void HGCalTower::addEtEm(double et) {
  etEm_ += et;
  addEt(et);
}

void HGCalTower::addEtHad(double et) {
  etHad_ += et;
  addEt(et);
}

void HGCalTower::addEt(double et) { this->setP4(PolarLorentzVector(this->pt() + et, this->eta(), this->phi(), 0.)); }

const HGCalTower& HGCalTower::operator+=(const HGCalTower& tower) {
  // NOTE: assume same eta and phi -> need an explicit check on the ID
  if (id().rawId() != tower.id().rawId()) {
    throw edm::Exception(edm::errors::StdException, "StdException")
        << "HGCalTower: adding to this tower with ID: " << id().rawId()
        << " one with different ID: " << tower.id().rawId() << std::endl;
  }
  addEt(tower.pt());
  etEm_ += tower.etEm();
  etHad_ += tower.etHad();

  return *this;
}

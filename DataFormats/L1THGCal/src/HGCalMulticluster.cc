#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

using namespace l1t;

HGCalMulticluster::HGCalMulticluster(const LorentzVector p4, int pt, int eta, int phi)
    : HGCalClusterT<l1t::HGCalCluster>(p4, pt, eta, phi), hOverE_(-99), hOverEValid_(false) {}

HGCalMulticluster::HGCalMulticluster(const edm::Ptr<l1t::HGCalCluster> &clusterSeed, float fraction)
    : HGCalClusterT<l1t::HGCalCluster>(clusterSeed, fraction), hOverE_(-99), hOverEValid_(false) {}

HGCalMulticluster::~HGCalMulticluster() {}

void HGCalMulticluster::saveEnergyInterpretation(const HGCalMulticluster::EnergyInterpretation eInt, double energy) {
  energyInterpretationFractions_[eInt] = energy / this->energy();
}

double HGCalMulticluster::interpretationFraction(const HGCalMulticluster::EnergyInterpretation eInt) const {
  auto intAndEnergyFraction = energyInterpretationFractions_.find(eInt);
  if (intAndEnergyFraction == energyInterpretationFractions_.end()) {
    // NOTE: this is an arbitary choice: we return the default cluster energy if this interpreation is not available!
    return 1;
  }
  return intAndEnergyFraction->second;
}

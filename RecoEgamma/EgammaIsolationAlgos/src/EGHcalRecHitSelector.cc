#include "RecoEgamma/EgammaIsolationAlgos/interface/EGHcalRecHitSelector.h"

#include <limits>

EGHcalRecHitSelector::EGHcalRecHitSelector(const edm::ParameterSet& config, edm::ConsumesCollector cc)
    : maxDIEta_(config.getParameter<int>("maxDIEta")),
      maxDIPhi_(config.getParameter<int>("maxDIPhi")),
      minEnergyHB_(config.getParameter<double>("minEnergyHB")),
      minEnergyHEDepth1_(config.getParameter<double>("minEnergyHEDepth1")),
      minEnergyHEDefault_(config.getParameter<double>("minEnergyHEDefault")),
      towerMapToken_(cc.esConsumes<CaloTowerConstituentsMap, CaloGeometryRecord, edm::Transition::BeginRun>()) {}

edm::ParameterSetDescription EGHcalRecHitSelector::makePSetDescription() {
  edm::ParameterSetDescription desc;
  desc.add<int>("maxDIEta", 5);
  desc.add<int>("maxDIPhi", 5);
  desc.add<double>("minEnergyHB", 0.8);
  desc.add<double>("minEnergyHEDepth1", 0.1);
  desc.add<double>("minEnergyHEDefault", 0.2);
  return desc;
}

int EGHcalRecHitSelector::calDIEta(int iEta1, int iEta2) {
  int dEta = iEta1 - iEta2;
  if (iEta1 * iEta2 < 0) {  //-ve to +ve transistion and no crystal at zero
    if (dEta < 0)
      dEta++;
    else
      dEta--;
  }
  return dEta;
}

int EGHcalRecHitSelector::calDIPhi(int iPhi1, int iPhi2) {
  int dPhi = iPhi1 - iPhi2;
  if (dPhi > 72 / 2)
    dPhi -= 72;
  else if (dPhi < -72 / 2)
    dPhi += 72;
  return dPhi;
}

float EGHcalRecHitSelector::getMinEnergyHCAL_(HcalDetId id) const {
  if (id.subdetId() == HcalBarrel)
    return minEnergyHB_;
  else if (id.subdetId() == HcalEndcap) {
    if (id.depth() == 1)
      return minEnergyHEDepth1_;
    else
      return minEnergyHEDefault_;
  } else
    return std::numeric_limits<float>::max();
}

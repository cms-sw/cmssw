#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

using namespace reco;

ElectronHcalHelper::ElectronHcalHelper(const Configuration& cfg, edm::ConsumesCollector&& cc) : cfg_(cfg) {
  if (cfg_.hOverEConeSize == 0) {
    return;
  }

  if (cfg_.useTowers) {
    hcalChannelQualityToken_ = cc.esConsumes(edm::ESInputTag("", "withTopo"));
    hcalTopologyToken_ = cc.esConsumes();
    towerMapToken_ = cc.esConsumes();
  } else {
    caloGeometryToken_ = cc.esConsumes();
  }
}

void ElectronHcalHelper::beginEvent(const edm::Event& evt, const edm::EventSetup& eventSetup) {
  if (cfg_.hOverEConeSize == 0) {
    return;
  }

  if (cfg_.useTowers) {
    towersFromCollection_ = &evt.get(cfg_.hcalTowers);

    towerMap_ = &eventSetup.getData(towerMapToken_);
    hcalQuality_ = &eventSetup.getData(hcalChannelQualityToken_);
    hcalTopology_ = &eventSetup.getData(hcalTopologyToken_);

    towerIso1_ =
        std::make_unique<EgammaTowerIsolation>(cfg_.hOverEConeSize, 0., cfg_.hOverEPtMin, 1, towersFromCollection_);
    towerIso2_ =
        std::make_unique<EgammaTowerIsolation>(cfg_.hOverEConeSize, 0., cfg_.hOverEPtMin, 2, towersFromCollection_);
  } else {
    hcalIso_ = std::make_unique<EgammaHcalIsolation>(cfg_.hOverEConeSize,
                                                     0.,
                                                     cfg_.hOverEHBMinE,
                                                     cfg_.hOverEHFMinE,
                                                     0.,
                                                     0.,
                                                     eventSetup.getHandle(caloGeometryToken_),
                                                     evt.get(cfg_.hcalRecHits));
  }
}

double ElectronHcalHelper::hcalESum(const SuperCluster& sc, const std::vector<CaloTowerDetId>* excludeTowers) const {
  if (cfg_.hOverEConeSize == 0) {
    return 0;
  }
  if (cfg_.useTowers) {
    return (hcalESumDepth1(sc, excludeTowers) + hcalESumDepth2(sc, excludeTowers));
  } else {
    return hcalIso_->getHcalESum(&sc);
  }
}

double ElectronHcalHelper::hcalESumDepth1(const SuperCluster& sc,
                                          const std::vector<CaloTowerDetId>* excludeTowers) const {
  if (cfg_.hOverEConeSize == 0) {
    return 0;
  }
  if (cfg_.useTowers) {
    return towerIso1_->getTowerESum(&sc, excludeTowers);
  } else {
    return hcalIso_->getHcalESumDepth1(&sc);
  }
}

double ElectronHcalHelper::hcalESumDepth2(const SuperCluster& sc,
                                          const std::vector<CaloTowerDetId>* excludeTowers) const {
  if (cfg_.hOverEConeSize == 0) {
    return 0;
  }
  if (cfg_.useTowers) {
    return towerIso2_->getTowerESum(&sc, excludeTowers);
  } else {
    return hcalIso_->getHcalESumDepth2(&sc);
  }
}

bool ElectronHcalHelper::hasActiveHcal(const reco::SuperCluster& sc) const {
  if (cfg_.checkHcalStatus && cfg_.hOverEConeSize != 0 && cfg_.useTowers) {
    return egamma::hasActiveHcal(egamma::towersOf(sc, *towerMap_), *towerMap_, *hcalQuality_, *hcalTopology_);
  } else {
    return true;
  }
}


#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHadTower.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;

ElectronHcalHelper::ElectronHcalHelper(const Configuration& cfg)
    : cfg_(cfg), caloGeomCacheId_(0), hcalIso_(nullptr), towerIso1_(nullptr), towerIso2_(nullptr), hadTower_(nullptr) {}

void ElectronHcalHelper::checkSetup(const edm::EventSetup& es) {
  if (cfg_.hOverEConeSize == 0) {
    return;
  }

  if (cfg_.useTowers) {
    delete hadTower_;
    hadTower_ = new EgammaHadTower(es);
  } else {
    unsigned long long newCaloGeomCacheId_ = es.get<CaloGeometryRecord>().cacheIdentifier();
    if (caloGeomCacheId_ != newCaloGeomCacheId_) {
      caloGeomCacheId_ = newCaloGeomCacheId_;
      es.get<CaloGeometryRecord>().get(caloGeom_);
    }
  }
}

void ElectronHcalHelper::readEvent(const edm::Event& evt) {
  if (cfg_.hOverEConeSize == 0) {
    return;
  }

  if (cfg_.useTowers) {
    delete towerIso1_;
    towerIso1_ = nullptr;
    delete towerIso2_;
    towerIso2_ = nullptr;

    towersFromCollection_ = &evt.get(cfg_.hcalTowers);
    towerIso1_ = new EgammaTowerIsolation(cfg_.hOverEConeSize, 0., cfg_.hOverEPtMin, 1, towersFromCollection_);
    towerIso2_ = new EgammaTowerIsolation(cfg_.hOverEConeSize, 0., cfg_.hOverEPtMin, 2, towersFromCollection_);
  } else {
    delete hcalIso_;
    hcalIso_ = nullptr;

    edm::Handle<HBHERecHitCollection> hbhe_;
    if (!evt.getByToken(cfg_.hcalRecHits, hbhe_)) {
      edm::LogError("ElectronHcalHelper::readEvent") << "failed to get the rechits";
    }

    hcalIso_ = new EgammaHcalIsolation(
        cfg_.hOverEConeSize, 0., cfg_.hOverEHBMinE, cfg_.hOverEHFMinE, 0., 0., caloGeom_, *hbhe_);
  }
}

std::vector<CaloTowerDetId> ElectronHcalHelper::hcalTowersBehindClusters(const reco::SuperCluster& sc) const {
  return hadTower_->towersOf(sc);
}

double ElectronHcalHelper::hcalESumDepth1BehindClusters(const std::vector<CaloTowerDetId>& towers) const {
  return hadTower_->getDepth1HcalESum(towers, *towersFromCollection_);
}

double ElectronHcalHelper::hcalESumDepth2BehindClusters(const std::vector<CaloTowerDetId>& towers) const {
  return hadTower_->getDepth2HcalESum(towers, *towersFromCollection_);
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
    return hadTower_->hasActiveHcal(hadTower_->towersOf(sc));
  } else {
    return true;
  }
}

ElectronHcalHelper::~ElectronHcalHelper() {
  if (cfg_.hOverEConeSize == 0) {
    return;
  }
  if (cfg_.useTowers) {
    delete towerIso1_;
    delete towerIso2_;
    delete hadTower_;
  } else {
    delete hcalIso_;
  }
}

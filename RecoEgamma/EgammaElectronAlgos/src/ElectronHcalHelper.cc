#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

using namespace reco;

ElectronHcalHelper::ElectronHcalHelper(const Configuration& cfg, edm::ConsumesCollector&& cc) : cfg_(cfg) {
  if (cfg_.hOverEConeSize == 0. and !cfg_.onlyBehindCluster) {
    return;
  }

  caloGeometryToken_ = cc.esConsumes();
  hcalTopologyToken_ = cc.esConsumes();
  hcalChannelQualityToken_ = cc.esConsumes(edm::ESInputTag("", "withTopo"));
  hcalSevLvlComputerToken_ = cc.esConsumes();
  towerMapToken_ = cc.esConsumes();
}

void ElectronHcalHelper::beginEvent(const edm::Event& evt, const edm::EventSetup& eventSetup) {
  if (cfg_.hOverEConeSize == 0. and !cfg_.onlyBehindCluster) {
    return;
  }

  hcalTopology_ = &eventSetup.getData(hcalTopologyToken_);
  hcalChannelQuality_ = &eventSetup.getData(hcalChannelQualityToken_);
  hcalSevLvlComputer_ = &eventSetup.getData(hcalSevLvlComputerToken_);
  towerMap_ = &eventSetup.getData(towerMapToken_);

  if (cfg_.onlyBehindCluster) {
    hcalIso_ = std::make_unique<EgammaHcalIsolation>(EgammaHcalIsolation::InclusionRule::isBehindClusterSeed,
                                                     0.,
                                                     EgammaHcalIsolation::InclusionRule::withinConeAroundCluster,
                                                     0.,
                                                     cfg_.eThresHB,
                                                     EgammaHcalIsolation::arrayHB{{0., 0., 0., 0.}},
                                                     cfg_.maxSeverityHB,
                                                     cfg_.eThresHE,
                                                     EgammaHcalIsolation::arrayHE{{0., 0., 0., 0., 0., 0., 0.}},
                                                     cfg_.maxSeverityHE,
                                                     evt.get(cfg_.hbheRecHits),
                                                     eventSetup.getHandle(caloGeometryToken_),
                                                     eventSetup.getHandle(hcalTopologyToken_),
                                                     eventSetup.getHandle(hcalChannelQualityToken_),
                                                     eventSetup.getHandle(hcalSevLvlComputerToken_),
                                                     towerMap_);
  } else {
    hcalIso_ = std::make_unique<EgammaHcalIsolation>(EgammaHcalIsolation::InclusionRule::withinConeAroundCluster,
                                                     cfg_.hOverEConeSize,
                                                     EgammaHcalIsolation::InclusionRule::withinConeAroundCluster,
                                                     0.,
                                                     cfg_.eThresHB,
                                                     EgammaHcalIsolation::arrayHB{{0., 0., 0., 0.}},
                                                     cfg_.maxSeverityHB,
                                                     cfg_.eThresHE,
                                                     EgammaHcalIsolation::arrayHE{{0., 0., 0., 0., 0., 0., 0.}},
                                                     cfg_.maxSeverityHE,
                                                     evt.get(cfg_.hbheRecHits),
                                                     eventSetup.getHandle(caloGeometryToken_),
                                                     eventSetup.getHandle(hcalTopologyToken_),
                                                     eventSetup.getHandle(hcalChannelQualityToken_),
                                                     eventSetup.getHandle(hcalSevLvlComputerToken_),
                                                     towerMap_);
  }
}

bool ElectronHcalHelper::hasActiveHcal(const reco::SuperCluster& sc) const {
  return (cfg_.checkHcalStatus)
             ? egamma::hasActiveHcal(hcalTowersBehindClusters(sc), *towerMap_, *hcalChannelQuality_, *hcalTopology_)
             : true;
}

double ElectronHcalHelper::hcalESum(const SuperCluster& sc, int depth) const {
  return (cfg_.onlyBehindCluster)     ? hcalIso_->getHcalESumBc(&sc, depth)
         : (cfg_.hOverEConeSize > 0.) ? hcalIso_->getHcalESum(&sc, depth)
                                      : 0.;
}

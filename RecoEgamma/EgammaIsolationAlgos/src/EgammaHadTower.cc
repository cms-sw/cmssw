#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHadTower.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include <algorithm>
#include <iostream>

//#define EDM_ML_DEBUG

EgammaHadTower::EgammaHadTower(const edm::EventSetup& es, HoeMode mode) : mode_(mode) {
  edm::ESHandle<CaloTowerConstituentsMap> ctmaph;
  es.get<CaloGeometryRecord>().get(ctmaph);
  towerMap_ = &(*ctmaph);
  nMaxClusters_ = 4;

  edm::ESHandle<HcalChannelQuality> hQuality;
  es.get<HcalChannelQualityRcd>().get("withTopo", hQuality);
  hcalQuality_ = hQuality.product();

  edm::ESHandle<HcalTopology> hcalTopology;
  es.get<HcalRecNumberingRecord>().get(hcalTopology);
  hcalTopology_ = hcalTopology.product();
}

CaloTowerDetId EgammaHadTower::towerOf(const reco::CaloCluster& cluster) const {
  DetId detid = cluster.seed();
  if (detid.det() != DetId::Ecal) {
    // Basic clusters of hybrid super-cluster do not have the seed set; take the first DetId instead
    // Should be checked . The single Tower Mode should be favoured until fixed
    detid = cluster.hitsAndFractions()[0].first;
    if (detid.det() != DetId::Ecal) {
      CaloTowerDetId tower;
      return tower;
    }
  }
  CaloTowerDetId id(towerMap_->towerOf(detid));
  return id;
}

std::vector<CaloTowerDetId> EgammaHadTower::towersOf(const reco::SuperCluster& sc) const {
  std::vector<CaloTowerDetId> towers;
  std::vector<reco::CaloClusterPtr> orderedClusters;

  // in this mode, check only the tower behind the seed
  if (mode_ == SingleTower) {
    towers.push_back(towerOf(*sc.seed()));
  }

  // in this mode check the towers behind each basic cluster
  if (mode_ == TowersBehindCluster) {
    // Loop on the basic clusters
    for (auto it = sc.clustersBegin(); it != sc.clustersEnd(); ++it) {
      orderedClusters.push_back(*it);
    }
    std::sort(orderedClusters.begin(), orderedClusters.end(), [](auto& c1, auto& c2) { return (*c1 > *c2); });
    unsigned nclusters = orderedClusters.size();
    for (unsigned iclus = 0; iclus < nclusters && iclus < nMaxClusters_; ++iclus) {
      // Get the tower
      CaloTowerDetId id = towerOf(*(orderedClusters[iclus]));
#ifdef EDM_ML_DEBUG
      std::cout << "CaloTowerId " << id << std::endl;
#endif
      if (std::find(towers.begin(), towers.end(), id) == towers.end()) {
        towers.push_back(id);
      }
    }
  }
  //  if(towers.size() > 4) {
  //    std::cout << " NTOWERS " << towers.size() << " ";
  //    for(unsigned i=0; i<towers.size() ; ++i) {
  //      std::cout << towers[i] << " ";
  //    }
  //    std::cout <<  std::endl;
  //    for ( unsigned iclus=0 ; iclus < orderedClusters.size(); ++iclus) {
  //      // Get the tower
  //      CaloTowerDetId id = towerOf(*(orderedClusters[iclus]));
  //      std::cout << " Pos " << orderedClusters[iclus]->position() << " " << orderedClusters[iclus]->energy() << " " << id ;
  //    }
  //    std::cout << std::endl;
  //  }
  return towers;
}

double EgammaHadTower::getDepth1HcalESum(const std::vector<CaloTowerDetId>& towers,
                                         CaloTowerCollection const& towerCollection) const {
  double esum = 0.;
  for (auto const& tower : towerCollection) {
    if (std::find(towers.begin(), towers.end(), tower.id()) != towers.end()) {
      esum += tower.ietaAbs() < 18 || tower.ietaAbs() > 29 ? tower.hadEnergy() : tower.hadEnergyHeInnerLayer();
    }
  }
  return esum;
}

double EgammaHadTower::getDepth2HcalESum(const std::vector<CaloTowerDetId>& towers,
                                         CaloTowerCollection const& towerCollection) const {
  double esum = 0.;
  for (auto const& tower : towerCollection) {
    if (std::find(towers.begin(), towers.end(), tower.id()) != towers.end()) {
      esum += tower.hadEnergyHeOuterLayer();
    }
  }
  return esum;
}

bool EgammaHadTower::hasActiveHcal(const std::vector<CaloTowerDetId>& towers) const {
  bool active = false;
  int statusMask = ((1 << HcalChannelStatus::HcalCellOff) | (1 << HcalChannelStatus::HcalCellMask) |
                    (1 << HcalChannelStatus::HcalCellDead));
#ifdef EDM_ML_DEBUG
  std::cout << "DEBUG: hasActiveHcal called with " << towers.size() << " detids. First tower detid ieta "
            << towers.front().ieta() << " iphi " << towers.front().iphi() << std::endl;
#endif
  for (auto towerid : towers) {
    unsigned int ngood = 0, nbad = 0;
    for (DetId id : towerMap_->constituentsOf(towerid)) {
      if (id.det() != DetId::Hcal) {
        continue;
      }
      HcalDetId hid(id);
      if (hid.subdet() != HcalBarrel && hid.subdet() != HcalEndcap)
        continue;
#ifdef EDM_ML_DEBUG
      std::cout << "EgammaHadTower DetId " << std::hex << id.rawId() << "  hid.rawId  " << hid.rawId() << std::dec
                << "   sub " << hid.subdet() << "   ieta " << hid.ieta() << "   iphi " << hid.iphi() << "   depth "
                << hid.depth() << std::endl;
#endif
      // Sunanda's fix for 2017 Plan1
      // and removed protection
      int status =
          hcalQuality_->getValues((DetId)(hcalTopology_->idFront(HcalDetId(id))), /*throwOnFail=*/true)->getValue();

#ifdef EDM_ML_DEBUG
      std::cout << "channels status = " << std::hex << status << std::dec << "  int value = " << status << std::endl;
#endif

      if (status & statusMask) {
#ifdef EDM_ML_DEBUG
        std::cout << "          BAD!" << std::endl;
#endif
        nbad++;
      } else {
        ngood++;
      }
    }
#ifdef EDM_ML_DEBUG
    std::cout << "    overall ngood " << ngood << " nbad " << nbad << "\n";
#endif
    if (nbad == 0 || (ngood > 0 && nbad < ngood)) {
      active = true;
    }
  }
  return active;
}

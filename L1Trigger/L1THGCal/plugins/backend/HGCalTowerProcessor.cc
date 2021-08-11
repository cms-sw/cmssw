#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTowerMap2DImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTowerMap3DImpl.h"

class HGCalTowerProcessor : public HGCalTowerProcessorBase {
public:
  HGCalTowerProcessor(const edm::ParameterSet& conf) : HGCalTowerProcessorBase(conf) {
    includeTrigCells_ = conf.getParameter<bool>("includeTrigCells"),
    towermap2D_ = std::make_unique<HGCalTowerMap2DImpl>(conf.getParameterSet("towermap_parameters"));
    towermap3D_ = std::make_unique<HGCalTowerMap3DImpl>();
  }

  void eventSetup(const edm::EventSetup& es) override { towermap2D_->eventSetup(es); }

  void run(const std::pair<edm::Handle<l1t::HGCalTowerMapBxCollection>, edm::Handle<l1t::HGCalClusterBxCollection>>&
               collHandle,
           l1t::HGCalTowerBxCollection& collTowers,
           const edm::EventSetup& es) override {
    es.get<CaloGeometryRecord>().get("", triggerGeometry_);

    auto& towerMapCollHandle = collHandle.first;
    auto& unclTCsCollHandle = collHandle.second;

    /* create a persistent vector of pointers to the towerMaps */
    std::vector<edm::Ptr<l1t::HGCalTowerMap>> towerMapsPtrs;
    for (unsigned i = 0; i < towerMapCollHandle->size(); ++i) {
      towerMapsPtrs.emplace_back(towerMapCollHandle, i);
    }

    if (includeTrigCells_) {
      /* create additional TowerMaps from the unclustered TCs */

      // translate our HGCalClusters into HGCalTriggerCells
      std::vector<edm::Ptr<l1t::HGCalTriggerCell>> trigCellVec;
      for (unsigned i = 0; i < unclTCsCollHandle->size(); ++i) {
        edm::Ptr<l1t::HGCalCluster> ptr(unclTCsCollHandle, i);
        for (const auto& itTC : ptr->constituents()) {
          trigCellVec.push_back(itTC.second);
        }
      }

      // fill the TowerMaps with the HGCalTriggersCells
      l1t::HGCalTowerMapBxCollection towerMapsFromUnclTCs;
      towermap2D_->buildTowerMap2D(trigCellVec, towerMapsFromUnclTCs);

      /* merge the two sets of TowerMaps */
      unsigned int towerMapsPtrsSize = towerMapsPtrs.size();
      for (unsigned int i = 0; i < towerMapsFromUnclTCs.size(); ++i) {
        towerMapsPtrs.emplace_back(&(towerMapsFromUnclTCs[i]), i + towerMapsPtrsSize);
      }

      /* call to towerMap3D clustering */
      towermap3D_->buildTowerMap3D(towerMapsPtrs, collTowers);
    } else {
      /* call to towerMap3D clustering */
      towermap3D_->buildTowerMap3D(towerMapsPtrs, collTowers);
    }
  }

private:
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
  bool includeTrigCells_;

  /* algorithms instances */
  std::unique_ptr<HGCalTowerMap2DImpl> towermap2D_;
  std::unique_ptr<HGCalTowerMap3DImpl> towermap3D_;
};

DEFINE_EDM_PLUGIN(HGCalTowerFactory, HGCalTowerProcessor, "HGCalTowerProcessor");

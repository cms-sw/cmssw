#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTowerMap2DImpl.h"

class HGCalTowerMapProcessor : public HGCalTowerMapProcessorBase {
public:
  HGCalTowerMapProcessor(const edm::ParameterSet& conf) : HGCalTowerMapProcessorBase(conf) {
    towermap2D_ = std::make_unique<HGCalTowerMap2DImpl>(conf.getParameterSet("towermap_parameters"));
  }

  void run(const edm::Handle<l1t::HGCalTriggerCellBxCollection>& collHandle,
           l1t::HGCalTowerMapBxCollection& collTowerMap,
           const edm::EventSetup& es) override {
    es.get<CaloGeometryRecord>().get("", triggerGeometry_);
    towermap2D_->eventSetup(es);

    /* create a persistent vector of pointers to the trigger-cells */
    std::vector<edm::Ptr<l1t::HGCalTriggerCell>> triggerCellsPtrs;
    for (unsigned i = 0; i < collHandle->size(); ++i) {
      edm::Ptr<l1t::HGCalTriggerCell> ptr(collHandle, i);
      triggerCellsPtrs.push_back(ptr);
    }

    /* call to towerMap2D clustering */
    towermap2D_->buildTowerMap2D(triggerCellsPtrs, collTowerMap);
  }

private:
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

  /* algorithms instances */
  std::unique_ptr<HGCalTowerMap2DImpl> towermap2D_;
};

DEFINE_EDM_PLUGIN(HGCalTowerMapFactory, HGCalTowerMapProcessor, "HGCalTowerMapProcessor");

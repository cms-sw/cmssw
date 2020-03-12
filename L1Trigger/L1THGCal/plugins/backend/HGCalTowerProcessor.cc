#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTowerMap3DImpl.h"

class HGCalTowerProcessor : public HGCalTowerProcessorBase {
public:
  HGCalTowerProcessor(const edm::ParameterSet& conf) : HGCalTowerProcessorBase(conf) {
    towermap3D_ = std::make_unique<HGCalTowerMap3DImpl>();
  }

  void run(const edm::Handle<l1t::HGCalTowerMapBxCollection>& collHandle,
           l1t::HGCalTowerBxCollection& collTowers,
           const edm::EventSetup& es) override {
    es.get<CaloGeometryRecord>().get("", triggerGeometry_);

    /* create a persistent vector of pointers to the towerMaps */
    std::vector<edm::Ptr<l1t::HGCalTowerMap>> towerMapsPtrs;
    for (unsigned i = 0; i < collHandle->size(); ++i) {
      edm::Ptr<l1t::HGCalTowerMap> ptr(collHandle, i);
      towerMapsPtrs.push_back(ptr);
    }

    /* call to towerMap3D clustering */
    towermap3D_->buildTowerMap3D(towerMapsPtrs, collTowers);
  }

private:
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

  /* algorithms instances */
  std::unique_ptr<HGCalTowerMap3DImpl> towermap3D_;
};

DEFINE_EDM_PLUGIN(HGCalTowerFactory, HGCalTowerProcessor, "HGCalTowerProcessor");

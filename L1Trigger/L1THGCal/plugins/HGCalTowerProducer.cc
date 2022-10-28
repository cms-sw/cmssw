#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include <memory>

class HGCalTowerProducer : public edm::stream::EDProducer<> {
public:
  HGCalTowerProducer(const edm::ParameterSet&);
  ~HGCalTowerProducer() override {}

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // inputs
  edm::EDGetToken input_towers_map_;
  edm::EDGetToken input_trigger_cells_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
  edm::ESGetToken<HGCalTriggerGeometryBase, CaloGeometryRecord> triggerGeomToken_;
  std::unique_ptr<HGCalTowerProcessorBase> towersProcess_;
};

DEFINE_FWK_MODULE(HGCalTowerProducer);

HGCalTowerProducer::HGCalTowerProducer(const edm::ParameterSet& conf)
    : input_towers_map_(consumes<l1t::HGCalTowerMapBxCollection>(conf.getParameter<edm::InputTag>("InputTowerMaps"))),
      input_trigger_cells_(
          consumes<l1t::HGCalClusterBxCollection>(conf.getParameter<edm::InputTag>("InputTriggerCells"))),
      triggerGeomToken_(esConsumes<HGCalTriggerGeometryBase, CaloGeometryRecord, edm::Transition::BeginRun>()) {
  //setup TowerMap parameters
  const edm::ParameterSet& towerParamConfig = conf.getParameterSet("ProcessorParameters");
  const std::string& towerProcessorName = towerParamConfig.getParameter<std::string>("ProcessorName");
  towersProcess_ = HGCalTowerFactory::get()->create(towerProcessorName, towerParamConfig);

  produces<l1t::HGCalTowerBxCollection>(towersProcess_->name());
}

void HGCalTowerProducer::beginRun(const edm::Run& /*run*/, const edm::EventSetup& es) {
  triggerGeometry_ = es.getHandle(triggerGeomToken_);
  towersProcess_->setGeometry(triggerGeometry_.product());
}

void HGCalTowerProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  // Output collections
  auto towers_output = std::make_unique<l1t::HGCalTowerBxCollection>();

  // Input collections
  std::pair<edm::Handle<l1t::HGCalTowerMapBxCollection>, edm::Handle<l1t::HGCalClusterBxCollection> > inputsColl;
  auto& towersMapBxColl = inputsColl.first;
  auto& clustersBxColl = inputsColl.second;

  e.getByToken(input_towers_map_, towersMapBxColl);
  e.getByToken(input_trigger_cells_, clustersBxColl);

  towersProcess_->run(inputsColl, *towers_output);

  e.put(std::move(towers_output), towersProcess_->name());
}

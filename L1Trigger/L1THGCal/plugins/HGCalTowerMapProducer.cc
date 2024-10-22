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

class HGCalTowerMapProducer : public edm::stream::EDProducer<> {
public:
  HGCalTowerMapProducer(const edm::ParameterSet&);
  ~HGCalTowerMapProducer() override {}

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // inputs
  edm::EDGetToken input_sums_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
  edm::ESGetToken<HGCalTriggerGeometryBase, CaloGeometryRecord> triggerGeomToken_;
  std::unique_ptr<HGCalTowerMapProcessorBase> towersMapProcess_;
};

DEFINE_FWK_MODULE(HGCalTowerMapProducer);

HGCalTowerMapProducer::HGCalTowerMapProducer(const edm::ParameterSet& conf)
    : input_sums_(consumes<l1t::HGCalTriggerSumsBxCollection>(conf.getParameter<edm::InputTag>("InputTriggerSums"))),
      triggerGeomToken_(esConsumes<HGCalTriggerGeometryBase, CaloGeometryRecord, edm::Transition::BeginRun>()) {
  //setup TowerMap parameters
  const edm::ParameterSet& towerMapParamConfig = conf.getParameterSet("ProcessorParameters");
  const std::string& towerMapProcessorName = towerMapParamConfig.getParameter<std::string>("ProcessorName");
  towersMapProcess_ = std::unique_ptr<HGCalTowerMapProcessorBase>{
      HGCalTowerMapFactory::get()->create(towerMapProcessorName, towerMapParamConfig)};

  produces<l1t::HGCalTowerMapBxCollection>(towersMapProcess_->name());
}

void HGCalTowerMapProducer::beginRun(const edm::Run& /*run*/, const edm::EventSetup& es) {
  triggerGeometry_ = es.getHandle(triggerGeomToken_);
  towersMapProcess_->setGeometry(triggerGeometry_.product());
}

void HGCalTowerMapProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  // Output collections
  auto towersMap_output = std::make_unique<l1t::HGCalTowerMapBxCollection>();

  // Input collections
  edm::Handle<l1t::HGCalTriggerSumsBxCollection> trigSumBxColl;

  e.getByToken(input_sums_, trigSumBxColl);

  towersMapProcess_->run(trigSumBxColl, *towersMap_output);

  e.put(std::move(towersMap_output), towersMapProcess_->name());
}

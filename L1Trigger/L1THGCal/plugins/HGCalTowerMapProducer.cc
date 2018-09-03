#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "L1Trigger/L1THGCal/interface/HGCalTowerMapProcessorBase.h"

#include <sstream>
#include <memory>

class HGCalTowerMapProducer : public edm::stream::EDProducer<> { 
 public:    
  HGCalTowerMapProducer(const edm::ParameterSet&);
  ~HGCalTowerMapProducer() override { }
  
  void beginRun(const edm::Run&, 
                        const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  
 private:
  // inputs
  edm::EDGetToken input_cell_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

  std::unique_ptr<HGCalTowerMapProcessorBase> towersMapProcess_;
};

DEFINE_FWK_MODULE(HGCalTowerMapProducer);

HGCalTowerMapProducer::
HGCalTowerMapProducer(const edm::ParameterSet& conf):
  input_cell_(consumes<l1t::HGCalTriggerCellBxCollection>(conf.getParameter<edm::InputTag>("InputTriggerCells")))
{ 
  //setup TowerMap parameters
  const edm::ParameterSet& towerMapParamConfig = conf.getParameterSet("ProcessorParameters");
  const std::string& towerMapProcessorName = towerMapParamConfig.getParameter<std::string>("ProcessorName");
  HGCalTowerMapProcessorBase* towerMapProc = HGCalTowerMapFactory::get()->create(towerMapProcessorName, towerMapParamConfig);
  towersMapProcess_.reset(towerMapProc);
  
  produces<l1t::HGCalTowerMapBxCollection>(towersMapProcess_->name());
}

void HGCalTowerMapProducer::beginRun(const edm::Run& /*run*/, 
                                          const edm::EventSetup& es) {
  es.get<CaloGeometryRecord>().get("",triggerGeometry_);
  towersMapProcess_->setGeometry(triggerGeometry_.product());
  
}

void HGCalTowerMapProducer::produce(edm::Event& e, const edm::EventSetup& es) {

  // Output collections
  std::unique_ptr<l1t::HGCalTowerMapBxCollection> towersMap_output( new l1t::HGCalTowerMapBxCollection );
  
  // Input collections
  edm::Handle<l1t::HGCalTriggerCellBxCollection> trigCellBxColl;
  
  e.getByToken(input_cell_, trigCellBxColl);

  towersMapProcess_->run(trigCellBxColl, *towersMap_output, es);

  e.put(std::move(towersMap_output), towersMapProcess_->name());

}

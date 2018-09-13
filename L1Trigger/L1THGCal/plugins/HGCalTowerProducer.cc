#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "L1Trigger/L1THGCal/interface/HGCalTowerProcessorBase.h"

#include <sstream>
#include <memory>

class HGCalTowerProducer : public edm::stream::EDProducer<> { 
 public:    
  HGCalTowerProducer(const edm::ParameterSet&);
  ~HGCalTowerProducer() override { }
  
  void beginRun(const edm::Run&, 
                        const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  
 private:
  // inputs
  edm::EDGetToken input_towers_map_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

  std::unique_ptr<HGCalTowerProcessorBase> towersProcess_;
};

DEFINE_FWK_MODULE(HGCalTowerProducer);

HGCalTowerProducer::
HGCalTowerProducer(const edm::ParameterSet& conf):
  input_towers_map_(consumes<l1t::HGCalTowerMapBxCollection>(conf.getParameter<edm::InputTag>("InputTriggerCells")))
{ 
  //setup TowerMap parameters
  const edm::ParameterSet& towerParamConfig = conf.getParameterSet("ProcessorParameters");
  const std::string& towerProcessorName = towerParamConfig.getParameter<std::string>("ProcessorName");
  HGCalTowerProcessorBase* towerProc = HGCalTowerFactory::get()->create(towerProcessorName, towerParamConfig);
  towersProcess_.reset(towerProc);
  
  produces<l1t::HGCalTowerBxCollection>(towersProcess_->name());
}

void HGCalTowerProducer::beginRun(const edm::Run& /*run*/, 
                                          const edm::EventSetup& es) {				  
  es.get<CaloGeometryRecord>().get("",triggerGeometry_);
  towersProcess_->setGeometry(triggerGeometry_.product());
}

void HGCalTowerProducer::produce(edm::Event& e, const edm::EventSetup& es) {

  // Output collections
  std::unique_ptr<l1t::HGCalTowerBxCollection> towers_output( new l1t::HGCalTowerBxCollection );
  
  // Input collections
  edm::Handle<l1t::HGCalTowerMapBxCollection> towersMapBxColl;
  
  e.getByToken(input_towers_map_, towersMapBxColl);
  
  towersProcess_->run(towersMapBxColl, *towers_output, es);
  
  e.put(std::move(towers_output), towersProcess_->name());
}

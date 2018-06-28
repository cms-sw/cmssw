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
  ~HGCalTowerProducer() { }
  
  virtual void beginRun(const edm::Run&, 
                        const edm::EventSetup&);
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  // inputs
  edm::EDGetToken input_cell_, input_sums_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

  std::unique_ptr<HGCalTowerProcessorBase> towerProcess_;
};

DEFINE_FWK_MODULE(HGCalTowerProducer);

HGCalTowerProducer::
HGCalTowerProducer(const edm::ParameterSet& conf):
  input_cell_(consumes<l1t::HGCalTriggerCellBxCollection>(conf.getParameter<edm::InputTag>("InputTriggerCells"))),
  input_sums_(consumes<l1t::HGCalTriggerSumsBxCollection>(conf.getParameter<edm::InputTag>("InputTriggerSums")))
{   
  produces<l1t::HGCalTowerBxCollection>(towerProcess_->name());
}

void HGCalTowerProducer::beginRun(const edm::Run& /*run*/, 
                                          const edm::EventSetup& es) {				  
  es.get<CaloGeometryRecord>().get("",triggerGeometry_);
  towerProcess_->setGeometry(triggerGeometry_.product());
}

void HGCalTowerProducer::produce(edm::Event& e, const edm::EventSetup& es) {

  // Output collections
  std::unique_ptr<l1t::HGCalTowerBxCollection> towers_output( new l1t::HGCalTowerBxCollection );
  
  // Input collections
  edm::Handle<l1t::HGCalTowerMapBxCollection> towerMapBxColl;
  
  e.getByToken(input_cell_, towerMapBxColl);

  const l1t::HGCalTowerMapBxCollection inputTowerMapBxColl = *towerMapBxColl;
  
  towerProcess_->run(towerMapBxColl, *towers_output, es);
  
  e.put(std::move(towers_output), towerProcess_->name());
}

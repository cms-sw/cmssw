#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "L1Trigger/L1THGCal/interface/HGCalConcentratorProcessorBase.h"

#include <sstream>
#include <memory>


class HGCalConcentratorProducer : public edm::stream::EDProducer<> {  
 public:    
  HGCalConcentratorProducer(const edm::ParameterSet&);
  ~HGCalConcentratorProducer() { }
  
  virtual void beginRun(const edm::Run&, 
                        const edm::EventSetup&);
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  // inputs
  edm::EDGetToken input_cell_, input_sums_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
  std::string choice_;
  
  std::unique_ptr<HGCalConcentratorProcessorBase> concentratorProcess_;
};

DEFINE_FWK_MODULE(HGCalConcentratorProducer);

HGCalConcentratorProducer::
HGCalConcentratorProducer(const edm::ParameterSet& conf):
  input_cell_(consumes<l1t::HGCalTriggerCellBxCollection>(conf.getParameter<edm::InputTag>("bxCollection"))),
  input_sums_(consumes<l1t::HGCalTriggerSumsBxCollection>(conf.getParameter<edm::InputTag>("bxCollection")))
{
  //setup Concentrator parameters
  const edm::ParameterSet& concParamConfig = conf.getParameterSet("Concentratorparam");
  const std::string& concProcessorName = concParamConfig.getParameter<std::string>("ConcProcessorName");
  choice_ = concParamConfig.getParameter<std::string>("Method");
  HGCalConcentratorProcessorBase* concProc = HGCalConcentratorFactory::get()->create(concProcessorName, concParamConfig);
  concentratorProcess_.reset(concProc);

  concentratorProcess_->setProduces(*this);

}

void HGCalConcentratorProducer::beginRun(const edm::Run& /*run*/, 
                                          const edm::EventSetup& es) {
  es.get<IdealGeometryRecord>().get(triggerGeometry_);
  
  concentratorProcess_->setGeometry(triggerGeometry_.product());

}

void HGCalConcentratorProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  
  edm::Handle<l1t::HGCalTriggerCellBxCollection> trigCellBxColl;
  edm::Handle<l1t::HGCalTriggerSumsBxCollection> trigSumsBxColl;

  e.getByToken(input_cell_,trigCellBxColl);
  e.getByToken(input_sums_,trigSumsBxColl);

  const l1t::HGCalTriggerCellBxCollection& trigCellColl = *trigCellBxColl;
  //const l1t::HGCalTriggerSumsBxCollection& trigSums = *trigSumsBxColl;
  		  
  std::unordered_map<uint32_t, l1t::HGCalTriggerCellBxCollection> tc_modules;
  for(const auto& trigCell : trigCellColl) {
    uint32_t module = triggerGeometry_->getModuleFromTriggerCell(trigCell.detId());
    auto itr_insert = tc_modules.emplace(module, l1t::HGCalTriggerCellBxCollection());
    itr_insert.first->second.push_back(0,trigCell); //bx=0
  }
  		
  concentratorProcess_->reset();
  if (choice_ == "bestChoiceSelect"){    
    for( const auto& module_trigcell : tc_modules ) {
      concentratorProcess_->bestChoiceSelect(module_trigcell.second);
    }      	
  }
  else if (choice_ == "thresholdSelect"){
    for( const auto& module_trigcell : tc_modules ) {
      concentratorProcess_->thresholdSelect(trigCellColl);
    }  
  }
  concentratorProcess_->putInEvent(e);
   
}

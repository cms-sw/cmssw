#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorProcessorSelection.h"
#include <limits>

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

DEFINE_EDM_PLUGIN(HGCalConcentratorFactory, 
        HGCalConcentratorProcessorSelection,
        "HGCalConcentratorProcessorSelection");

HGCalConcentratorProcessorSelection::HGCalConcentratorProcessorSelection(const edm::ParameterSet& conf)  : 
  HGCalConcentratorProcessorBase(conf),
  choice_(conf.getParameter<std::string>("Method")),
  concentratorProcImpl_(conf)
{ 
}

void HGCalConcentratorProcessorSelection::run(const edm::Handle<l1t::HGCalTriggerCellBxCollection>& triggerCellCollInput, 
                                              l1t::HGCalTriggerCellBxCollection& triggerCellCollOutput,
                                              const edm::EventSetup& es)
{ 
  const l1t::HGCalTriggerCellBxCollection& collInput = *triggerCellCollInput;

  std::unordered_map<uint32_t, std::vector<l1t::HGCalTriggerCell>> tc_modules;
  for(const auto& trigCell : collInput) {
    uint32_t module = geometry_->getModuleFromTriggerCell(trigCell.detId());
    auto itr_insert = tc_modules.emplace(module, std::vector<l1t::HGCalTriggerCell>());
    itr_insert.first->second.push_back(trigCell); //bx=0
  }

  if (choice_ == "thresholdSelect")
  {
    for( const auto& module_trigcell : tc_modules ) {
      std::vector<l1t::HGCalTriggerCell> trigCellVecOutput;
      concentratorProcImpl_.thresholdSelectImpl(module_trigcell.second, trigCellVecOutput);
      // Push trigger Cells for each module from std::vector<l1t::HGCalTriggerCell> into the final collection
      for( auto trigCell = trigCellVecOutput.begin(); trigCell != trigCellVecOutput.end(); ++trigCell){
        triggerCellCollOutput.push_back(0, *trigCell);     
      }
    }
  }
  else if (choice_ == "bestChoiceSelect"){
    for( const auto& module_trigcell : tc_modules ) {  
      std::vector<l1t::HGCalTriggerCell> trigCellVecOutput;
      concentratorProcImpl_.bestChoiceSelectImpl(module_trigcell.second, trigCellVecOutput);
      
      // Push trigger Cells for each module from std::vector<l1t::HGCalTriggerCell> into the final collection
      for( auto trigCell = trigCellVecOutput.begin(); trigCell != trigCellVecOutput.end(); ++trigCell){
        triggerCellCollOutput.push_back(0, *trigCell);       
      }
    }
  }

}

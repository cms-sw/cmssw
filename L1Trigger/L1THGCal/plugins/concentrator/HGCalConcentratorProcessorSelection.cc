#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorProcessorSelection.h"
#include <limits>

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

DEFINE_EDM_PLUGIN(HGCalConcentratorFactory, 
        HGCalConcentratorProcessorSelection,
        "HGCalConcentratorProcessorSelection");

HGCalConcentratorProcessorSelection::HGCalConcentratorProcessorSelection(const edm::ParameterSet& conf)  : 
  HGCalConcentratorProcessorBase(conf),
  choice_(conf.getParameter<std::string>("Method")),
  concentratorProcImpl_(conf),
  triggercell_threshold_silicon_( conf.getParameter<double>("triggercell_threshold_silicon") ),
  triggercell_threshold_scintillator_( conf.getParameter<double>("triggercell_threshold_scintillator") )
{ 
}

void HGCalConcentratorProcessorSelection::runTriggCell(const l1t::HGCalTriggerCellBxCollection& triggerCellCollInput, 
                                                       l1t::HGCalTriggerCellBxCollection& triggerCellCollOutput,
                                                       const edm::EventSetup& es)
{    
  std::unordered_map<uint32_t, l1t::HGCalTriggerCellBxCollection> tc_modules;
  for(const auto& trigCell : triggerCellCollInput) {
    uint32_t module = geometry_->getModuleFromTriggerCell(trigCell.detId());
    auto itr_insert = tc_modules.emplace(module, l1t::HGCalTriggerCellBxCollection());
    itr_insert.first->second.push_back(0,trigCell); //bx=0
  }

  if (choice_ == "thresholdSelect")
  {
    for( const auto& module_trigcell : tc_modules ) {
    
      // Convert vector to collection
      std::vector<l1t::HGCalTriggerCell> trigCellVec;     
      trigCellVec = triggerTools_.collectionToVector(module_trigcell.second);
          
      concentratorProcImpl_.thresholdSelectImpl(trigCellVec);
      
      // Push trigger Cells for each module from std::vector<l1t::HGCalTriggerCell> into the final collection
      for( auto trigCell = trigCellVec.begin(); trigCell != trigCellVec.end(); ++trigCell){
        double triggercell_threshold = (trigCell->subdetId()==HGCHEB ? triggercell_threshold_scintillator_ : triggercell_threshold_silicon_);
        if(trigCell->mipPt()<triggercell_threshold) continue;
        triggerCellCollOutput.push_back(0, *trigCell);     
      }
    }
  }
  else if (choice_ == "bestChoiceSelect"){
    for( const auto& module_trigcell : tc_modules ) {
          
      // Convert vector to collection
      std::vector<l1t::HGCalTriggerCell> trigCellVec;
      trigCellVec = triggerTools_.collectionToVector(module_trigcell.second);

      concentratorProcImpl_.bestChoiceSelectImpl(trigCellVec);
      
      // Push trigger Cells for each module from std::vector<l1t::HGCalTriggerCell> into the final collection
      for( auto trigCell = trigCellVec.begin(); trigCell != trigCellVec.end(); ++trigCell){
        double triggercell_threshold = (trigCell->subdetId()==HGCHEB ? triggercell_threshold_scintillator_ : triggercell_threshold_silicon_);
        if(trigCell->mipPt()<triggercell_threshold) continue;
        triggerCellCollOutput.push_back(0, *trigCell);       
      }
    }
  }
}

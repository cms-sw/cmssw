#ifndef __L1Trigger_L1THGCal_HGCalConcentratorProcessorSelection_h__
#define __L1Trigger_L1THGCal_HGCalConcentratorProcessorSelection_h__

#include "L1Trigger/L1THGCal/interface/HGCalConcentratorProcessorBase.h"
#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorSelectionImpl.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"

class HGCalConcentratorProcessorSelection : public HGCalConcentratorProcessorBase 
{ 

  public:
    HGCalConcentratorProcessorSelection(const edm::ParameterSet& conf);
  
    void run(const edm::Handle<l1t::HGCalTriggerCellBxCollection>& triggerCellCollInput, l1t::HGCalTriggerCellBxCollection& triggerCellCollOutput, const edm::EventSetup& es) override;

  private:
    std::string choice_;
    
    HGCalConcentratorSelectionImpl concentratorProcImpl_;
     
    HGCalTriggerTools triggerTools_;

};

#endif

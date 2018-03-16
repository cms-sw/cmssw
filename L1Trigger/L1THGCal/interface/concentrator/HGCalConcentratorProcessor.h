#ifndef __L1Trigger_L1THGCal_HGCalConcentratorProcessor_h__
#define __L1Trigger_L1THGCal_HGCalConcentratorProcessor_h__

#include "L1Trigger/L1THGCal/interface/HGCalConcentratorProcessorBase.h"
#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorSelectionImpl.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"

class HGCalConcentratorProcessor : public HGCalConcentratorProcessorBase 
{

 public:
  HGCalConcentratorProcessor(const edm::ParameterSet& conf);
  
  virtual void setProduces(edm::stream::EDProducer<>& prod) const override final 
  {
    prod.produces<l1t::HGCalTriggerCellBxCollection>(name());
    prod.produces<l1t::HGCalTriggerSumsBxCollection>(name());
  }

  virtual void reset() override final 
  { 
    triggerCellConc_product_.reset( new l1t::HGCalTriggerCellBxCollection );
    triggerSumsConc_product_.reset( new l1t::HGCalTriggerSumsBxCollection );
  }

  void putInEvent(edm::Event& evt);
  
  void bestChoiceSelect(const l1t::HGCalTriggerCellBxCollection& coll);
  void thresholdSelect(const l1t::HGCalTriggerCellBxCollection& coll);
  
  std::vector<l1t::HGCalTriggerCell> trigCellCollectionToVector(int ibx, const l1t::HGCalTriggerCellBxCollection& coll);
  void trigCellVectorToCollection(int ibx, const std::vector<l1t::HGCalTriggerCell>);
  
  
 private:
  HGCalConcentratorSelectionImpl ConcentratorProcImpl_;
  
  std::unique_ptr<l1t::HGCalTriggerCellBxCollection> triggerCellConc_product_;
  std::unique_ptr<l1t::HGCalTriggerSumsBxCollection> triggerSumsConc_product_;

};

#endif

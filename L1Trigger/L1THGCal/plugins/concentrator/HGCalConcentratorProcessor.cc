#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorProcessor.h"
#include <limits>

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

DEFINE_EDM_PLUGIN(HGCalConcentratorFactory, 
        HGCalConcentratorProcessor,
        "HGCalConcentratorProcessor");

HGCalConcentratorProcessor::HGCalConcentratorProcessor(const edm::ParameterSet& conf)  : 
	HGCalConcentratorProcessorBase(conf),
	ConcentratorProcImpl_(conf)	
{ 
}

void HGCalConcentratorProcessor::putInEvent(edm::Event& evt)
{ 
  evt.put(std::move(triggerCellConc_product_), name());
  evt.put(std::move(triggerSumsConc_product_), name());
}

std::vector<l1t::HGCalTriggerCell> HGCalConcentratorProcessor::trigCellCollectionToVector(int ibx, const l1t::HGCalTriggerCellBxCollection& coll){
    
    std::vector<l1t::HGCalTriggerCell> trigCellVec;
        
    //loop over HGCalTriggerCellBxCollection for a given bx and put the HGCalTriggerCell objects into a std::vector
    for( std::vector<l1t::HGCalTriggerCell>::const_iterator it = coll.begin(ibx) ; it != coll.end(ibx) ; ++it )
    { 
      trigCellVec.push_back(*it); 
    }
 
    return trigCellVec;
}

// Transform std::vector<l1t::HGCalTriggerCell> to BXVector<HGCalTriggerCell> HGCalTriggerCellBxCollection
void HGCalConcentratorProcessor::trigCellVectorToCollection(int ibx, const std::vector<l1t::HGCalTriggerCell> trigCellVec)
{  
  for( auto trigCell = trigCellVec.begin(); trigCell != trigCellVec.end(); ++trigCell){
      triggerCellConc_product_->push_back(ibx, *trigCell);
  }    
   
}

void HGCalConcentratorProcessor::bestChoiceSelect(const l1t::HGCalTriggerCellBxCollection& coll)
{ 
  int bxFirst = coll.getFirstBX();
  int bxLast = coll.getLastBX();
     
  // loop over BX
  for(int ibx = bxFirst; ibx < bxLast+1; ++ibx) {
    
    std::vector<l1t::HGCalTriggerCell> trigCellVec;
    
    // Convert vector to collection
    trigCellVec = trigCellCollectionToVector(ibx, coll);
    
    // Selection in implementation class
    ConcentratorProcImpl_.bestChoiceSelectImpl(trigCellVec);
    
    // Convert std::vector to collection (BXVector), fill into the final product
    trigCellVectorToCollection(ibx, trigCellVec);    
  }
   

}

void 
HGCalConcentratorProcessor::
thresholdSelect(const l1t::HGCalTriggerCellBxCollection& coll)
{  
  int bxFirst = coll.getFirstBX();
  int bxLast = coll.getLastBX();
    
  // loop over BX
  for(int ibx = bxFirst; ibx < bxLast+1; ++ibx) {
  
    std::vector<l1t::HGCalTriggerCell> trigCellVec;
    
    // Convert vector to collection
    trigCellVec = trigCellCollectionToVector(ibx, coll);
    
    // Select in the implementation class
    ConcentratorProcImpl_.thresholdSelectImpl(trigCellVec);
  
    // Convert std::vector to collection (BXVector), fill into the final product 
    trigCellVectorToCollection(ibx, trigCellVec);    
  }  
}

#ifndef __L1Trigger_L1THGCal_HGCalConcentratorSelectionImpl_h__
#define __L1Trigger_L1THGCal_HGCalConcentratorSelectionImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"


#include <array>
#include <vector>

class HGCalConcentratorSelectionImpl
{

 public:
   HGCalConcentratorSelectionImpl(const edm::ParameterSet& conf);
  
   void bestChoiceSelectImpl(std::vector<l1t::HGCalTriggerCell>& trigCellVec);

   void thresholdSelectImpl(std::vector<l1t::HGCalTriggerCell>& trigCellVec);

   // Retrieve parameters
   size_t   nCellsInModule() const {return nCellsInModule_;}
   double linLSB() const {return linLSB_;}
   size_t nData() const {return nData_;}

   // Retrieve parameters
   int TCThreshold_ADC() const {return TCThreshold_ADC_;} 
   double TCThreshold_fC() const {return TCThreshold_fC_;} 

 private:
   size_t nData_;
   size_t nCellsInModule_;
   double linLSB_;
   int TCThreshold_ADC_;
   double TCThreshold_fC_;
    
};

#endif

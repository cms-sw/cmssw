#ifndef __L1Trigger_L1THGCal_HGCalVFEProcessorSums_h__
#define __L1Trigger_L1THGCal_HGCalVFEProcessorSums_h__

#include "L1Trigger/L1THGCal/interface/HGCalVFEProcessorBase.h"

#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFELinearizationImpl.h"
#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFESummationImpl.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalTriggerCellCalibration.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"


class HGCalVFEProcessorSums : public HGCalVFEProcessorBase
{
    
  public:
    
    HGCalVFEProcessorSums(const edm::ParameterSet& conf);
    
    void run(const HGCEEDigiCollection& ee,
                      const HGCHEDigiCollection& fh,
                      const HGCBHDigiCollection& bh, 
                      l1t::HGCalTriggerCellBxCollection& triggerCellColl, 
                      const edm::EventSetup& es) override;
	             
  private:
          
    HGCalVFELinearizationImpl vfeLinearizationImpl_;
    HGCalVFESummationImpl vfeSummationImpl_; 
    HGCalTriggerCellCalibration calibration_;

};    
    
#endif

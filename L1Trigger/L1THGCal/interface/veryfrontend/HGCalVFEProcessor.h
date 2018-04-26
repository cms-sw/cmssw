#ifndef __L1Trigger_L1THGCal_HGCalVFEProcessor_h__
#define __L1Trigger_L1THGCal_HGCalVFEProcessor_h__

#include "L1Trigger/L1THGCal/interface/HGCalVFEProcessorBase.h"

#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFELinearizationImpl.h"
#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFESummationImpl.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalTriggerCellCalibration.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"


class HGCalVFEProcessor : public HGCalVFEProcessorBase
{
    
  public:
    
    HGCalVFEProcessor(const edm::ParameterSet& conf);
    
    void vfeProcessing(const HGCEEDigiCollection& ee,
                	   const HGCHEDigiCollection& fh,
                	   const HGCBHDigiCollection& bh, const edm::EventSetup& es);
    
    void putInEvent(edm::Event& evt);
    
    virtual void setProduces(edm::stream::EDProducer<>& prod) const override final 
    { 
      prod.produces<l1t::HGCalTriggerCellBxCollection>("calibratedTriggerCells");
      prod.produces<l1t::HGCalTriggerSumsBxCollection>("calibratedTriggerCells");
    }
    
    virtual void reset() override final 
    {
      triggerCell_product_.reset( new l1t::HGCalTriggerCellBxCollection );
      triggerSums_product_.reset( new l1t::HGCalTriggerSumsBxCollection );
    }
             
    typedef std::unique_ptr<HGCalVFEProcessorBase> vfeProcessing_ptr;
        
  private:
          
    HGCalVFELinearizationImpl vfeLinearizationImpl_;
    HGCalVFESummationImpl vfeSummationImpl_; 
    HGCalTriggerCellCalibration calibration_;
    
    std::vector<l1t::HGCalTriggerCell> vecTrigCell_;
    std::unique_ptr<l1t::HGCalTriggerCellBxCollection> triggerCell_product_;
    std::unique_ptr<l1t::HGCalTriggerSumsBxCollection> triggerSums_product_;

    /* Parameters for calibration */ 
    double triggercell_threshold_silicon_;
    double triggercell_threshold_scintillator_;

};    
    
#endif

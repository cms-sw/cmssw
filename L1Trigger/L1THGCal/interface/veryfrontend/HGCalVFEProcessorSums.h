#ifndef __L1Trigger_L1THGCal_HGCalVFEProcessorSums_h__
#define __L1Trigger_L1THGCal_HGCalVFEProcessorSums_h__

#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFELinearizationImpl.h"
#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFESummationImpl.h"
#include "L1Trigger/L1THGCal/interface/HGCalVFECompressionImpl.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerCellCalibration.h"

class HGCalVFEProcessorSums : public HGCalVFEProcessorBase {
public:
  HGCalVFEProcessorSums(const edm::ParameterSet& conf);

  void run(const HGCalDigiCollection& digiColl,
           l1t::HGCalTriggerCellBxCollection& triggerCellColl,
           const edm::EventSetup& es) override;

private:
  std::unique_ptr<HGCalVFELinearizationImpl> vfeLinearizationImpl_;
  std::unique_ptr<HGCalVFESummationImpl> vfeSummationImpl_;
  std::unique_ptr<HGCalVFECompressionImpl> vfeCompressionImpl_;
  std::unique_ptr<HGCalTriggerCellCalibration> calibration_;
};

#endif

#ifndef __L1Trigger_L1THGCal_HGCalBackendStage1Processor_h__
#define __L1Trigger_L1THGCal_HGCalBackendStage1Processor_h__

#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalAlgoWrapperBase.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

#include "L1Trigger/L1THGCal/interface/backend/HGCalStage1TruncationImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalClusteringDummyImpl.h"

class HGCalBackendStage1Processor : public HGCalBackendStage1ProcessorBase {
public:
  HGCalBackendStage1Processor(const edm::ParameterSet& conf);

  void run(const std::pair<uint32_t, std::vector<edm::Ptr<l1t::HGCalTriggerCell>>>& fpga_id_tcs,
           std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& truncated_tcs) override;

private:
  std::unique_ptr<HGCalStage1TruncationWrapperBase> truncationWrapper_;
  const edm::ParameterSet conf_;
};

#endif

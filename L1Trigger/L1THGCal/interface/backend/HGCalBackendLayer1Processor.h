#ifndef __L1Trigger_L1THGCal_HGCalBackendLayer1Processor_h__
#define __L1Trigger_L1THGCal_HGCalBackendLayer1Processor_h__

#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

#include "L1Trigger/L1THGCal/interface/backend/HGCalStage1TruncationImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalClusteringDummyImpl.h"

class HGCalBackendLayer1Processor : public HGCalBackendLayer1ProcessorBase {
public:
  HGCalBackendLayer1Processor(const edm::ParameterSet& conf);

  void run(const edm::Handle<l1t::HGCalTriggerCellBxCollection>& collHandle,
           l1t::HGCalClusterBxCollection& collCluster2D) override;

private:
  std::unique_ptr<HGCalClusteringDummyImpl> clusteringDummy_;
  std::unique_ptr<HGCalStage1TruncationImpl> truncation_;
};

#endif

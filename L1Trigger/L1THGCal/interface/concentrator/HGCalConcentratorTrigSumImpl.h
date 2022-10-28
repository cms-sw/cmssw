#ifndef __L1Trigger_L1THGCal_HGCalConcentratorTrigSumImpl_h__
#define __L1Trigger_L1THGCal_HGCalConcentratorTrigSumImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include <vector>

class HGCalConcentratorTrigSumImpl {
public:
  HGCalConcentratorTrigSumImpl(const edm::ParameterSet& conf);

  void doSum(uint32_t module_id,
             const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
             std::vector<l1t::HGCalTriggerSums>& trigSumsVecOutput) const;

  void setGeometry(const HGCalTriggerGeometryBase* const geom) { triggerTools_.setGeometry(geom); }

private:
  HGCalTriggerTools triggerTools_;
};

#endif

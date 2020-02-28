#ifndef __L1Trigger_L1THGCal_HGCalVFELinearizationImpl_h__
#define __L1Trigger_L1THGCal_HGCalVFELinearizationImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include <array>
#include <vector>

class HGCalVFELinearizationImpl {
public:
  HGCalVFELinearizationImpl(const edm::ParameterSet& conf);

  void linearize(const std::vector<HGCDataFrame<DetId, HGCSample>>&, std::vector<std::pair<DetId, uint32_t>>&);

  // Retrieve parameters
  uint32_t linnBits() const { return linnBits_; }

private:
  double adcLSB_si_;
  double linLSB_si_;
  double adcsaturation_si_;
  uint32_t tdcnBits_si_;
  double tdcOnset_si_;
  uint32_t adcnBits_si_;
  double tdcsaturation_si_;
  double tdcLSB_si_;
  //
  double adcLSB_sc_;
  double linLSB_sc_;
  double adcsaturation_sc_;
  uint32_t tdcnBits_sc_;
  double tdcOnset_sc_;
  uint32_t adcnBits_sc_;
  double tdcsaturation_sc_;
  double tdcLSB_sc_;
  //
  uint32_t linMax_;
  uint32_t linnBits_;
};

#endif

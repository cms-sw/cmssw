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
  double adcLSB_;
  double linLSB_;
  double adcsaturation_;
  uint32_t tdcnBits_;
  double tdcOnsetfC_;
  uint32_t adcnBits_;
  double tdcsaturation_;
  uint32_t linnBits_;
  double tdcLSB_;
  uint32_t linMax_;
};

#endif

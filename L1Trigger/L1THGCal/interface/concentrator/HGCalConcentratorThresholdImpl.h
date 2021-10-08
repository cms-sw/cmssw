#ifndef __L1Trigger_L1THGCal_HGCalConcentratorThresholdImpl_h__
#define __L1Trigger_L1THGCal_HGCalConcentratorThresholdImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include <vector>

class HGCalConcentratorThresholdImpl {
public:
  HGCalConcentratorThresholdImpl(const edm::ParameterSet& conf);

  void select(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
              std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput,
              std::vector<l1t::HGCalTriggerCell>& trigCellVecNotSelected);

  void setGeometry(const HGCalTriggerGeometryBase* const geom) { triggerTools_.setGeometry(geom); }

private:
  double threshold_silicon_;
  double threshold_scintillator_;

  HGCalTriggerTools triggerTools_;
};

#endif
